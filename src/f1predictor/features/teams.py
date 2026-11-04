"""Team lineages: linking a team's old names to its current one.

Jolpica gives a team a new constructor_id whenever it rebrands, e.g. Toro
Rosso -> AlphaTauri -> RB. It's the same factory, staff and (mostly) car
concept, but without this mapping every history feature that spans seasons
would treat RB in 2024 as a brand-new team with no past.

Each old id maps to the id the team uses today. Ids not listed are their own
lineage (Ferrari is just Ferrari; Cadillac is genuinely new in 2026).

Used only where team history crosses seasons (team record at a track, the
team DNF rate). Rebrands happen between seasons, so "this season" features
don't need it.
"""

TEAM_LINEAGE = {
    "toro_rosso": "rb",          # Toro Rosso (to 2019) -> AlphaTauri (2020-23) -> RB (2024+)
    "alphatauri": "rb",
    "renault": "alpine",         # Renault (to 2020) -> Alpine (2021+)
    "force_india": "aston_martin",   # Force India (2018) -> Racing Point (2019-20) -> Aston Martin (2021+)
    "racing_point": "aston_martin",
    "sauber": "audi",            # Sauber (2018, 2024-25) / Alfa Romeo (2019-23) -> Audi (2026+)
    "alfa": "audi",
}


# Teams that are genuinely NEW (no predecessor), confirmed by a person. With
# TEAM_LINEAGE this classifies every team id: when Jolpica shows a team id
# for the first time, it must be one or the other, or the validation check
# validate/checks.check_team_ids warns (architecture review, 2026-11-03).
# Without that, a rebrand nobody added to TEAM_LINEAGE would silently start
# the team's track and DNF history from zero.
NEW_TEAMS = {
    "cadillac",                  # 2026: a new entry, no predecessor
}


def lineage(constructor_id: str) -> str:
    """The team's current id: lineage('toro_rosso') -> 'rb', lineage('ferrari') -> 'ferrari'."""
    return TEAM_LINEAGE.get(constructor_id, constructor_id)
