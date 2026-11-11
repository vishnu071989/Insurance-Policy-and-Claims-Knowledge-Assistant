ROLES = {"claims_handler", "policy_servicing", "customer_support"}

def validate_role(role: str) -> str:
    if role not in ROLES:
        raise PermissionError(f"Unknown role '{role}'. Allowed: {sorted(ROLES)}")
    return role
