from typing import List, Dict
from fractions import Fraction

power_mod = {0: 1.0, 1: 1.25, 2: 1.4, 3: 1.5, 4: 1.6, 5: 1.7, 6: 1.8}


def calc_attack(attack_stat: int, gear: int, sync_grid_additions: int):
    atk = attack_stat + gear + sync_grid_additions
    return atk


def cal_defence(
    base_move_dmg: int,
    effective_damage: int,
    modifier_stage_up: int,
    attack_stat: int,
    gear: int,
    sync_grid_additions: int,
) -> None:
    roll = 1.0
    crit_mod = 1.5
    attack = calc_attack(attack_stat, gear, sync_grid_additions)
    defence_low = ((base_move_dmg * attack) + 1) / (
        effective_damage / (power_mod[modifier_stage_up] * crit_mod) / roll
    )
    defence_high = ((base_move_dmg * attack) + 1) / (
        effective_damage / (power_mod[modifier_stage_up] * crit_mod) / (roll * 1.5)
    )
    print(f"This is the possible defence stat: {int(defence_low)}-{int(defence_high)}")


def TE_damage(
    move_name,
    base_power_of_move,
    modifier_stage_up,
    attack_stat,
    gear,
    sg_adds,
    defence,
    crit_mod=1.0,
):
    damage = (
        base_power_of_move * (calc_attack(attack_stat, gear, sg_adds) / defence) + 1
    ) * (power_mod[modifier_stage_up] * crit_mod)
    print(
        f"{move_name.capitalize()}, with a base power of {base_power_of_move} will do {int(damage*0.9)}-{int(damage*1.0)} damage against a Pokemon with {defence} defence points. If {move_name.capitalize()} was super-effective, it would {int(damage*0.9*2)}-{int(damage*1.0*2)} damage."
    )


cal_defence(
    base_move_dmg=186,
    effective_damage=10797,
    modifier_stage_up=6,
    attack_stat=380,
    gear=40,
    sync_grid_additions=10,
)
TE_damage(
    "Blast Burn",
    base_power_of_move=187,
    modifier_stage_up=6,
    attack_stat=380,
    gear=40,
    sg_adds=10,
    defence=20,
    crit_mod=1.5,
)
 