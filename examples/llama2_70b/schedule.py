"""Partial user mapping; DefaultMapper fills every unspecified decision."""

from aloi.mapping.schedule import schedule

with schedule("attention.wq") as q_schedule:
    q_schedule.tile("in_feature", factor=256)
    q_schedule.bind("out_feature", "channel")
    q_schedule.place("weight", "pim")


with schedule("feed_forward.w1") as gate_schedule:
    gate_schedule.tile("in_feature", factor=256)
    gate_schedule.pipeline("load", "mac", "read")
