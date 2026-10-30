"""Tests for the GIS / Ward Detection & Spatial Intelligence module (Part 10).

Covers, against the live (dev) database and real PostGIS:

* reverse geocoding — Nominatim success + graceful degradation on
  rate-limit (429), transport error, and timeouts (no real network: the
  database-backed lookup tests rely on injected ``httpx.MockTransport``).
* ward detection — point-in-polygon returns the containing operational ward for
  a Pune point (inside the reference WARD-1 Kondhwa polygon) and ``None`` for
  coordinates outside every boundary.
* nearby places & critical infrastructure — ``ST_DWithin`` geography-distance
  lookup returns roads/hospitals/schools/bus stops ordered by distance and the
  radius is clamped to ``GIS_MAX_RADIUS_M``. When the DB has no rows for a
  category the service falls back to LIVE OSM data (Overpass), and stays empty
  on any remote failure / when disabled ("Nearby infrastructure data
  unavailable").
* distance math — pure great-circle unit tests.
* API surface + RBAC — authenticated users (any role) may call lookup/wards/
  distance; unauthenticated requests get 401; out-of-range coordinates → 400.
* ward labeling — the four reference boundaries are surfaced with
  ``is_demo=False`` + their administrative geography (city/state/country); a
  facility row created with ``is_demo=True`` is labelled demo.
"""

import uuid

import httpx
import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import create_access_token
from app.db.session import async_session_factory
from app.models import CriticalLocation, User
from app.models.enums import CriticalLocationCategory
from app.schemas.auth import RegisterIn
from app.schemas.geo import GeoPlace, ReverseGeocodeOut
from app.services import auth_service
from app.services.geo_service import GeoService, _osm_category_for, get_geo_service
from tests.helpers import any_active_ward_id

_PASSWORD = "TestPass#2026"
_BASE = "/api/v1/geo"
_SETTINGS = get_settings()

# A real Pune point inside the reference WARD-1 (Kondhwa) operational polygon.
_SEED_LAT = 18.4634
_SEED_LON = 73.8912

# A point guaranteed to hold no registry rows, for tests that exercise the
# genuinely-empty-database behavior (≈500 km from every Pune facility).
_REMOTE_LAT = 17.4327
_REMOTE_LON = 78.3885


def _unique_email(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@example.com"


def _isolate_registry(monkeypatch) -> None:
    """Force the registry-aware pipeline to look empty for deterministic tests.

    Part 35 makes ``geo_lookup``/``find_nearby_places`` resolve against the
    verified facility registry first. The real dev database legitimately holds
    ingested Pune facilities, so tests that exercise the *live fallback* must
    stub registry coverage away (class-level, because ``geo_lookup`` mints its
    own ``InfrastructureRegistry`` internally).
    """
    from app.services.infrastructure_registry import InfrastructureRegistry

    async def empty_counts(self, db, categories):
        return {}

    async def empty_nearby(self, db, latitude, longitude, radius_m, categories, limit):
        return {}

    monkeypatch.setattr(InfrastructureRegistry, "_registry_counts", empty_counts)
    monkeypatch.setattr(InfrastructureRegistry, "_nearby_from_registry", empty_nearby)


async def _citizen_token(email: str) -> str:
    async with async_session_factory() as db:
        await auth_service.register_user(
            db,
            RegisterIn(
                email=email,
                password=_PASSWORD,
                full_name="GIS Citizen",
                ward_id=await any_active_ward_id(db),
            ),
        )
        user = await db.scalar(select(User).where(User.email == email))
    return create_access_token(str(user.id), "CITIZEN")


async def _delete_user(email: str) -> None:
    async with async_session_factory() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if user is not None:
            await db.delete(user)
            await db.commit()


async def _add_critical_location(
    name: str, category: CriticalLocationCategory, lat: float, lon: float, is_demo: bool
) -> uuid.UUID:
    from sqlalchemy import func

    async with async_session_factory() as db:
        row = CriticalLocation(
            name=name,
            category=category,
            latitude=lat,
            longitude=lon,
            address="Test facility",
            is_demo=is_demo,
            geom=func.ST_SetSRID(func.ST_MakePoint(lon, lat), 4326),
        )
        db.add(row)
        await db.flush()
        row_id = row.id
        await db.commit()
        return row_id


async def _delete_critical_locations(*ids: uuid.UUID) -> None:
    async with async_session_factory() as db:
        for loc_id in ids:
            row = await db.get(CriticalLocation, loc_id)
            if row is not None:
                await db.delete(row)
        await db.commit()


# --------------------------------------------------------------------------- #
# Distance math (pure, no DB)
# --------------------------------------------------------------------------- #
def test_calculate_distance_equator_degree_degrees():
    svc = get_geo_service()
    # 1 degree of latitude ≈ 111.2 km.
    d = svc.calculate_distance(0.0, 0.0, 1.0, 0.0)
    assert pytest.approx(d, rel=0.01) == 111_195.0


def test_calculate_distance_same_point_is_zero():
    svc = get_geo_service()
    assert svc.calculate_distance(_SEED_LAT, _SEED_LON, _SEED_LAT, _SEED_LON) == 0.0


def test_calculate_distance_symmetric():
    svc = get_geo_service()
    a = svc.calculate_distance(_SEED_LAT, _SEED_LON, 18.5665, 73.9122)
    b = svc.calculate_distance(18.5665, 73.9122, _SEED_LAT, _SEED_LON)
    assert a == pytest.approx(b)


def test_calculate_distance_known_city_pair():
    # Mumbai (19.076, 72.8777) → Pune (18.5204, 73.8567) ≈ 120 km.
    svc = get_geo_service()
    d = svc.calculate_distance(19.0760, 72.8777, 18.5204, 73.8567)
    assert pytest.approx(d / 1000.0, rel=0.05) == 120.0


# --------------------------------------------------------------------------- #
# Ward detection (live PostGIS)
# --------------------------------------------------------------------------- #
async def test_find_ward_detects_operational_ward_at_seed_area():
    svc = get_geo_service()
    async with async_session_factory() as db:
        ward = await svc.find_ward(db, _SEED_LAT, _SEED_LON)
    assert ward is not None
    assert ward.code == "WARD-1"
    assert ward.name == "Ward 1 — Kondhwa"
    assert ward.city == "Pune"
    assert ward.state == "Maharashtra"
    assert ward.country == "India"
    assert ward.is_demo is False


async def test_find_ward_none_outside_supported_region():
    svc = get_geo_service()
    async with async_session_factory() as db:
        # Well outside every Pune operational polygon.
        assert await svc.find_ward(db, 60.0, 30.0) is None
        assert await svc.find_ward(db, 17.4327, 78.3885) is None  # Hyderabad
        assert await svc.find_ward(db, 18.5307, 73.8439) is None  # Shivajinagar gap


async def test_find_ward_invalid_coordinates_raises():
    svc = get_geo_service()
    async with async_session_factory() as db:
        with pytest.raises(Exception):
            await svc.find_ward(db, 105.0, 0.0)  # lat out of range


# --------------------------------------------------------------------------- #
# Nearby places & critical infrastructure (live PostGIS + OSM fallback)
# --------------------------------------------------------------------------- #
async def test_find_nearby_places_returns_created_facilities():
    svc = get_geo_service()
    ids = [
        await _add_critical_location(
            "Pune Civic Hospital", CriticalLocationCategory.HOSPITAL, _SEED_LAT, _SEED_LON, True
        ),
        await _add_critical_location(
            "Kondhwa Vidyalaya",
            CriticalLocationCategory.SCHOOL,
            _SEED_LAT + 0.002,
            _SEED_LON,
            True,
        ),
        await _add_critical_location(
            "Kondhwa Bus Stop",
            CriticalLocationCategory.BUS_STOP,
            _SEED_LAT,
            _SEED_LON + 0.003,
            False,
        ),
    ]
    try:
        async with async_session_factory() as db:
            places = await svc.find_nearby_places(
                db,
                _SEED_LAT,
                _SEED_LON,
                _SETTINGS.GIS_CRITICAL_RADIUS_M,
                limit=1000,
            )
        names = {p.name for p in places}
        # The live registry legitimately holds real facilities near the seed, so
        # a generous limit guarantees the three test rows are within the result.
        assert {"Pune Civic Hospital", "Kondhwa Vidyalaya", "Kondhwa Bus Stop"} <= names
        hospital = next(p for p in places if p.name == "Pune Civic Hospital")
        assert hospital.is_demo is True
        bus_stop = next(p for p in places if p.name == "Kondhwa Bus Stop")
        assert bus_stop.is_demo is False
    finally:
        await _delete_critical_locations(*ids)


async def test_find_nearby_places_filters_by_category():
    svc = get_geo_service()
    hospital_id = await _add_critical_location(
        "Only Pune Hospital", CriticalLocationCategory.HOSPITAL, _SEED_LAT, _SEED_LON, True
    )
    school_id = await _add_critical_location(
        "Not A Hospital", CriticalLocationCategory.SCHOOL, _SEED_LAT, _SEED_LON, True
    )
    try:
        async with async_session_factory() as db:
            hospitals = await svc.find_nearby_places(
                db, _SEED_LAT, _SEED_LON, 5000.0, category=CriticalLocationCategory.HOSPITAL
            )
        names = {p.name for p in hospitals}
        assert "Only Pune Hospital" in names
        assert "Not A Hospital" not in names
    finally:
        await _delete_critical_locations(hospital_id, school_id)


async def test_find_nearby_places_radius_is_clamped():
    svc = get_geo_service()
    huge = 50_000_000.0  # far above GIS_MAX_RADIUS_M
    near_id = await _add_critical_location(
        "Nearby Clinic", CriticalLocationCategory.HOSPITAL, _SEED_LAT, _SEED_LON, False
    )
    far_id = await _add_critical_location(
        "Far Clinic", CriticalLocationCategory.HOSPITAL, _SEED_LAT + 0.9, _SEED_LON, False
    )
    try:
        async with async_session_factory() as db:
            places = await svc.find_nearby_places(db, _SEED_LAT, _SEED_LON, huge)
        names = {p.name for p in places}
        assert "Nearby Clinic" in names
        assert "Far Clinic" not in names
        # Every returned place must be within the *clamped* max radius.
        for p in places:
            assert (p.distance_m or 0.0) <= _SETTINGS.GIS_MAX_RADIUS_M + 1.0
    finally:
        await _delete_critical_locations(near_id, far_id)


async def test_no_facilities_with_overpass_disabled_returns_empty():
    # conftest disables live Overpass by default → no registry rows near the
    # lookup point means the lookup returns nothing (the UI shows "Nearby
    # infrastructure data unavailable"). The remote point avoids the real Pune
    # facilities now legitimately held in the live registry table.
    svc = get_geo_service()
    async with async_session_factory() as db:
        places = await svc.find_nearby_places(
            db, _REMOTE_LAT, _REMOTE_LON, 2000.0, category=CriticalLocationCategory.SCHOOL
        )
    assert places == []


async def test_overpass_fallback_used_when_db_empty(monkeypatch):
    svc = get_geo_service()
    calls: list[CriticalLocationCategory | None] = []

    async def fake_overpass(lat, lon, radius, category, limit=20, client=None):
        calls.append(category)
        return [
            GeoPlace(
                id=None,
                name="Live Pune School",
                category=category,
                latitude=lat,
                longitude=lon,
                distance_m=100.0,
                is_demo=False,
            )
        ]

    monkeypatch.setattr(svc, "_overpass_find", fake_overpass)
    async with async_session_factory() as db:
        places = await svc.find_nearby_places(
            db, _REMOTE_LAT, _REMOTE_LON, 2000.0, category=CriticalLocationCategory.SCHOOL
        )
    assert places and places[0].name == "Live Pune School"
    assert places[0].is_demo is False
    assert calls == [CriticalLocationCategory.SCHOOL]


def _overpass_payload() -> dict:
    return {
        "elements": [
            {
                "type": "node",
                "id": 1,
                "lat": _SEED_LAT,
                "lon": _SEED_LON,
                "tags": {"name": "Om Hospital", "addr:street": "Kondhwa Rd"},
            },
            {
                "type": "way",
                "id": 2,
                "center": {"lat": _SEED_LAT + 0.002, "lon": _SEED_LON + 0.002},
                "tags": {"name": "Pune Railway Station"},
            },
        ]
    }


async def test_overpass_request_parses_live_elements(monkeypatch):
    svc = get_geo_service()
    monkeypatch.setattr(svc._settings, "GIS_OVERPASS_ENABLED", True)
    monkeypatch.setattr(svc._settings, "GIS_CACHE_ENABLED", False)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert "data=" in str(request.url)  # QL sent as a GET param, not POST
        assert "around" in str(request.url)  # distance + point encoded
        return httpx.Response(200, json=_overpass_payload())

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=_SETTINGS.GIS_OVERPASS_URL)
    places = await svc._overpass_find(
        _SEED_LAT, _SEED_LON, 2000.0, CriticalLocationCategory.HOSPITAL, client=client
    )
    assert len(places) == 2
    assert {p.name for p in places} == {"Om Hospital", "Pune Railway Station"}
    assert all(p.is_demo is False for p in places)
    assert all(p.category == CriticalLocationCategory.HOSPITAL for p in places)
    assert all(p.distance_m is not None for p in places)


async def test_overpass_failure_returns_empty_list(monkeypatch):
    svc = get_geo_service()
    monkeypatch.setattr(svc._settings, "GIS_OVERPASS_ENABLED", True)
    monkeypatch.setattr(svc._settings, "GIS_CACHE_ENABLED", False)

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("overpass unreachable")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=_SETTINGS.GIS_OVERPASS_URL)
    places = await svc._overpass_find(
        _SEED_LAT, _SEED_LON, 2000.0, CriticalLocationCategory.SCHOOL, client=client
    )
    assert places == []


async def test_overpass_empty_elements_is_a_success(monkeypatch):
    # An Overpass 200 with zero elements means "genuinely no facilities" — it
    # must be reported as ok=True (so the UI shows the empty state, not the
    # temporarily-unavailable state). The empty result is also cacheable.
    svc = get_geo_service()
    monkeypatch.setattr(svc._settings, "GIS_OVERPASS_ENABLED", True)
    monkeypatch.setattr(svc._settings, "GIS_CACHE_ENABLED", False)

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"elements": []})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=_SETTINGS.GIS_OVERPASS_URL)
    places, ok, source, cached = await svc._fetch_overpass(
        _SEED_LAT, _SEED_LON, 2000.0, CriticalLocationCategory.BUS_STOP, client=client
    )
    assert places == []
    assert ok is True
    expected_host = httpx.URL(svc._overpass_endpoints()[0]).host
    assert source == f"overpass:{expected_host}"
    assert cached is False


async def test_overpass_http_error_then_mirror_succeeds(monkeypatch):
    # Primary returns HTTP 504 (busy) and the mirror serves the data.
    svc = get_geo_service()
    monkeypatch.setattr(svc._settings, "GIS_OVERPASS_ENABLED", True)
    monkeypatch.setattr(svc._settings, "GIS_CACHE_ENABLED", False)

    async def handler(request: httpx.Request) -> httpx.Response:
        if "overpass-api.de" in str(request.url):
            return httpx.Response(504, text="server busy")
        return httpx.Response(200, json=_overpass_payload())

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=_SETTINGS.GIS_OVERPASS_URL)
    places, ok, source, _cached = await svc._fetch_overpass(
        _SEED_LAT, _SEED_LON, 2000.0, CriticalLocationCategory.HOSPITAL, client=client
    )
    assert ok is True
    assert len(places) == 2
    assert "maps.mail.ru" in (source or "")


def test_osm_category_rules_resolve_combined_fetch():
    samples = [
        ({"amenity": "hospital"}, CriticalLocationCategory.HOSPITAL),
        ({"healthcare": "clinic"}, CriticalLocationCategory.HOSPITAL),
        ({"amenity": "school"}, CriticalLocationCategory.SCHOOL),
        ({"amenity": "kindergarten"}, CriticalLocationCategory.SCHOOL),
        ({"amenity": "bus_station"}, CriticalLocationCategory.BUS_STOP),
        ({"highway": "bus_stop"}, CriticalLocationCategory.BUS_STOP),
        ({"public_transport": "platform"}, CriticalLocationCategory.BUS_STOP),
        ({"amenity": "police"}, CriticalLocationCategory.POLICE_STATION),
        ({"amenity": "fire_station"}, CriticalLocationCategory.FIRE_STATION),
        ({"railway": "station"}, CriticalLocationCategory.TRANSPORT),
        ({"amenity": "library"}, CriticalLocationCategory.PUBLIC_FACILITY),
        ({"amenity": "marketplace"}, CriticalLocationCategory.PUBLIC_FACILITY),
        ({"amenity": "place_of_worship"}, CriticalLocationCategory.PUBLIC_FACILITY),
        ({"office": "government"}, CriticalLocationCategory.GOVERNMENT_BUILDING),
        ({"building": "government"}, CriticalLocationCategory.GOVERNMENT_BUILDING),
        ({"amenity": "townhall"}, CriticalLocationCategory.GOVERNMENT_BUILDING),
        ({"highway": "primary"}, CriticalLocationCategory.ROAD),
        ({"amenity": "cafe"}, CriticalLocationCategory.OTHER),
    ]
    for tags, expected in samples:
        assert _osm_category_for(tags) == expected


def test_infra_cache_key_rounds_coordinates_and_namespaces():
    svc = get_geo_service()
    key = svc._cache_key_for(CriticalLocationCategory.HOSPITAL, 18.4633599, 73.8912401, 500.0)
    assert ":nearby:hospital:" in key
    assert "18.46336,73.89124:500" in key
    key_all = svc._cache_key_for(None, 18.4633599, 73.8912401, 500.0)
    assert "nearby:all:" in key_all


async def test_geo_lookup_with_live_data_populates_all_categories(monkeypatch):
    # With an empty verified registry the full lookup surfaces REAL (Overpass)
    # facilities for every category and reports nearby_status="available".
    svc = get_geo_service()
    _isolate_registry(monkeypatch)

    async def fake_addr(lat, lon, client=None):
        return ReverseGeocodeOut(degraded=False, source="test")

    async def fake_ward(db, lat, lon):
        return None

    async def fake_fetch(lat, lon, radius, category, limit=20, client=None):
        return (
            [
                GeoPlace(
                    id=None,
                    name=f"Live {category.value}",
                    category=category,
                    latitude=lat,
                    longitude=lon,
                    distance_m=90.0,
                    is_demo=False,
                )
            ],
            True,
            "overpass:test",
            False,
        )

    monkeypatch.setattr(svc, "reverse_geocode", fake_addr)
    monkeypatch.setattr(svc, "find_ward", fake_ward)
    monkeypatch.setattr(svc, "_fetch_overpass", fake_fetch)
    async with async_session_factory() as db:
        out = await svc.geo_lookup(db, _SEED_LAT, _SEED_LON)

    assert out.nearby_status == "available"
    assert len(out.hospitals) == 1
    assert len(out.schools) == 1
    assert len(out.bus_stops) == 1
    assert len(out.police_stations) == 1
    assert len(out.fire_stations) == 1
    assert len(out.public_facilities) == 1
    assert len(out.government_buildings) == 1
    # Critical union covers every POI facility category.
    union_categories = {p.category for p in out.critical_infrastructure}
    assert CriticalLocationCategory.HOSPITAL in union_categories
    assert CriticalLocationCategory.POLICE_STATION in union_categories
    assert CriticalLocationCategory.PUBLIC_FACILITY in union_categories
    assert CriticalLocationCategory.GOVERNMENT_BUILDING in union_categories


async def test_geo_lookup_nearby_status_unavailable_when_live_down(monkeypatch):
    # Empty registry + live Overpass unavailable ⇒ "no facilities" is NOT claimed.
    svc = get_geo_service()
    _isolate_registry(monkeypatch)
    monkeypatch.setattr(svc._settings, "GIS_OVERPASS_ENABLED", False)

    async def fake_addr(lat, lon, client=None):
        return ReverseGeocodeOut(degraded=False, source="test")

    async def fake_ward(db, lat, lon):
        return None

    monkeypatch.setattr(svc, "reverse_geocode", fake_addr)
    monkeypatch.setattr(svc, "find_ward", fake_ward)
    async with async_session_factory() as db:
        out = await svc.geo_lookup(db, _SEED_LAT, _SEED_LON)

    assert out.nearby_status == "unavailable"
    assert out.hospitals == []
    assert out.schools == []
    assert out.bus_stops == []
    assert out.police_stations == []


async def test_geo_lookup_nearby_status_empty_when_live_ok_but_no_data(monkeypatch):
    # Every lookup succeeded but nothing exists within range ⇒ genuinely empty.
    svc = get_geo_service()
    _isolate_registry(monkeypatch)

    async def fake_addr(lat, lon, client=None):
        return ReverseGeocodeOut(degraded=False, source="test")

    async def fake_ward(db, lat, lon):
        return None

    async def fake_fetch(lat, lon, radius, category, limit=20, client=None):
        return [], True, "overpass:test", False

    monkeypatch.setattr(svc, "reverse_geocode", fake_addr)
    monkeypatch.setattr(svc, "find_ward", fake_ward)
    monkeypatch.setattr(svc, "_fetch_overpass", fake_fetch)
    async with async_session_factory() as db:
        out = await svc.geo_lookup(db, _SEED_LAT, _SEED_LON)

    assert out.nearby_status == "empty"
    assert out.critical_infrastructure == []


# --------------------------------------------------------------------------- #
# Reverse geocoding (injected httpx transport — no network)
# --------------------------------------------------------------------------- #
def _svc_with_client(handler) -> tuple[GeoService, httpx.AsyncClient]:
    svc = get_geo_service()
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url=_SETTINGS.GIS_BASE_URL)
    return svc, client


async def test_reverse_geocode_success():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"display_name": "7, Kondhwa Main Road, Kondhwa, Pune, Maharashtra"},
        )

    svc, client = _svc_with_client(handler)
    out = await svc.reverse_geocode(_SEED_LAT, _SEED_LON, client)
    assert out.degraded is False
    assert out.source == "nominatim"
    assert "Kondhwa Main Road" in (out.address or "")


async def test_reverse_geocode_rate_limit_degrades():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "Rate Limited"})

    svc, client = _svc_with_client(handler)
    out = await svc.reverse_geocode(_SEED_LAT, _SEED_LON, client)
    assert out.degraded is True
    assert out.address is None


async def test_reverse_geocode_transport_error_degrades():
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connection timed out")

    svc, client = _svc_with_client(handler)
    out = await svc.reverse_geocode(_SEED_LAT, _SEED_LON, client)
    assert out.degraded is True
    assert out.address is None


async def test_reverse_geocode_server_error_degrades():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    svc, client = _svc_with_client(handler)
    out = await svc.reverse_geocode(_SEED_LAT, _SEED_LON, client)
    assert out.degraded is True
    assert out.address is None


# --------------------------------------------------------------------------- #
# API surface + RBAC
# --------------------------------------------------------------------------- #
async def test_lookup_requires_auth(client):
    r = await client.post(f"{_BASE}/lookup", json={"latitude": _SEED_LAT, "longitude": _SEED_LON})
    assert r.status_code == 401


async def test_wards_requires_auth(client):
    r = await client.get(f"{_BASE}/wards")
    assert r.status_code == 401


async def test_distance_requires_auth(client):
    r = await client.post(
        f"{_BASE}/distance",
        json={"latitude_a": 1.0, "longitude_a": 2.0, "latitude_b": 3.0, "longitude_b": 4.0},
    )
    assert r.status_code == 401


async def test_lookup_valid_auth_returns_ward_payload(client):
    email = _unique_email("geo-lookup")
    token = await _citizen_token(email)
    try:
        r = await client.post(
            f"{_BASE}/lookup",
            json={"latitude": _SEED_LAT, "longitude": _SEED_LON, "radius_m": 5000.0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["ward"]["code"] == "WARD-1"
        assert data["ward"]["is_demo"] is False
        assert data["ward"]["city"] == "Pune"
        assert data["demo_label"] == "DEMO DATA"
        assert data["address"]["source"] == "nominatim"
        # The verified registry may legitimately hold real ingested Pune
        # facilities near this point; the contract is that every infra key is a
        # list (and demo data is never fabricated).
        for key in ("hospitals", "schools", "bus_stops", "nearby_roads", "critical_infrastructure"):
            assert isinstance(data[key], list)
    finally:
        await _delete_user(email)


async def test_lookup_accepts_any_authenticated_role(client):
    from app.core.security import hash_password
    from app.models import Role
    from app.models.enums import RoleName

    email = _unique_email("geo-officer")
    async with async_session_factory() as db:
        role = await db.scalar(select(Role).where(Role.name == RoleName.OFFICER.value))
        officer = User(
            email=email,
            full_name="GIS Officer",
            password_hash=hash_password(_PASSWORD),
            role_id=role.id,
            is_active=True,
            is_email_verified=True,
        )
        db.add(officer)
        await db.flush()
        officer_id = officer.id
        await db.commit()
    token = create_access_token(str(officer_id), RoleName.OFFICER.value)
    try:
        r = await client.get(f"{_BASE}/wards", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        assert {"wards"} <= set(r.json())
    finally:
        await _delete_user(email)


async def test_distance_endpoint_pure_math(client):
    email = _unique_email("geo-dist")
    token = await _citizen_token(email)
    try:
        r = await client.post(
            f"{_BASE}/distance",
            json={"latitude_a": 0.0, "longitude_a": 0.0, "latitude_b": 1.0, "longitude_b": 0.0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert pytest.approx(data["distance_km"], rel=0.01) == 111.195
        assert data["distance_m"] > 0
    finally:
        await _delete_user(email)


async def test_lookup_out_of_range_coordinates_returns_422(client):
    email = _unique_email("geo-bad")
    token = await _citizen_token(email)
    try:
        r = await client.post(
            f"{_BASE}/lookup",
            json={"latitude": 120.0, "longitude": 0.0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 422, r.text
    finally:
        await _delete_user(email)


def test_validate_coordinates_rejects_invalid():
    svc = get_geo_service()
    for lat, lon in [(120.0, 0.0), (0.0, 200.0), (-91.0, 0.0), (0.0, -181.0)]:
        with pytest.raises(Exception):
            svc.validate_coordinates(lat, lon)
    # Valid values pass without raising.
    svc.validate_coordinates(0.0, 0.0)


async def test_wards_returns_pune_boundary_rings(client):
    email = _unique_email("geo-wards")
    token = await _citizen_token(email)
    try:
        r = await client.get(f"{_BASE}/wards", headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
        data = r.json()
        wards = data["wards"]
        assert len(wards) >= 4
        by_code = {w["code"]: w for w in wards}
        for code in ("WARD-1", "WARD-2", "WARD-3", "WARD-4"):
            ward = by_code[code]
            assert ward["is_demo"] is False
            assert ward["city"] == "Pune"
            assert ward["state"] == "Maharashtra"
            assert ward["country"] == "India"
            assert "geometry" in ward and len(ward["geometry"]) >= 5  # closed ring
            assert "centroid" in ward and len(ward["centroid"]) == 2
    finally:
        await _delete_user(email)
