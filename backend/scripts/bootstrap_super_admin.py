"""Create the first production Super Admin interactively.

Run from the backend directory with:
    python scripts/bootstrap_super_admin.py
"""

from __future__ import annotations

import asyncio
from getpass import getpass

from pydantic import EmailStr, TypeAdapter
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password, is_strong_password
from app.db.session import async_session_factory, engine
from app.models import Role, User, UserProfile
from app.models.enums import RoleName
from app.services.audit_service import ACTION_CREATE, record_audit

_BOOTSTRAP_LOCK_ID = 731_492_681_103


async def _create_super_admin(
    db: AsyncSession,
    *,
    email: str,
    full_name: str,
    password: str,
) -> None:
    await db.execute(
        text("SELECT pg_advisory_xact_lock(:lock_id)"),
        {"lock_id": _BOOTSTRAP_LOCK_ID},
    )

    existing_admin = await db.scalar(
        select(User.id)
        .join(Role, User.role_id == Role.id)
        .where(Role.name == RoleName.SUPER_ADMIN.value)
        .limit(1)
    )
    if existing_admin is not None:
        raise RuntimeError("A SUPER_ADMIN account already exists; refusing to create another.")

    existing_user = await db.scalar(select(User.id).where(User.email == email))
    if existing_user is not None:
        raise RuntimeError("That email already belongs to an account; no changes were made.")

    role = await db.scalar(select(Role).where(Role.name == RoleName.SUPER_ADMIN.value))
    if role is None:
        role = Role(
            name=RoleName.SUPER_ADMIN.value,
            description="Super administrator with panel management rights.",
        )
        db.add(role)
        await db.flush()
    elif not role.is_active:
        raise RuntimeError("The SUPER_ADMIN role is inactive; refusing to change it.")

    user = User(
        email=email,
        password_hash=hash_password(password),
        full_name=full_name,
        role_id=role.id,
        is_active=True,
        is_email_verified=True,
    )
    db.add(user)
    await db.flush()
    db.add(UserProfile(user_id=user.id))
    await record_audit(
        db,
        actor_id=None,
        action=ACTION_CREATE,
        entity_type="user",
        entity_id=str(user.id),
        after={"email": user.email, "role": role.name, "source": "one-time-bootstrap"},
    )


async def main() -> None:
    email = str(TypeAdapter(EmailStr).validate_python(input("Initial Super Admin email: ").strip()))
    full_name = input("Full name: ").strip()
    if not full_name or len(full_name) > 255:
        raise ValueError("Full name is required and must be at most 255 characters.")

    password = getpass("New unique password (input hidden): ")
    valid, reason = is_strong_password(password)
    if not valid:
        raise ValueError(reason or "Password does not meet the strength requirements.")
    if password != getpass("Confirm password (input hidden): "):
        raise ValueError("Passwords do not match.")

    async with async_session_factory() as db:
        async with db.begin():
            await _create_super_admin(
                db,
                email=email.lower(),
                full_name=full_name,
                password=password,
            )

    print(f"Created the initial SUPER_ADMIN account for {email.lower()}.")
    print("Sign in, verify access, then store the credentials in your password manager.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
