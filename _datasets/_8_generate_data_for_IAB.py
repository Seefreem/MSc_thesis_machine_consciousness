#!/usr/bin/env python3
"""
Text-level inattentional blindness experiment simulator.

Scene: a classroom starts empty, students (male/female) come in and leave,
sometimes handing in homework. A distractor (teacher/cleaner/dog) appears
midway for a short duration.

Primary task (for the model later): count how many male/female students handed in.
Secondary task: detect whether teacher/cleaner/dog entered.

This script generates:
- per-timestep textual "statistical statements"
- a final summary dict
"""

from __future__ import annotations

import json
import random
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


# -----------------------------
# Constants / helpers
# -----------------------------
GENDERS = ("female", "male")

ACTION_COME_IN = "coming in"
ACTION_LEAVE = "leaving"
ACTION_SIT = "sitting down"
ACTION_HAND_IN = "handing in"
ACTION_NOTHING = "doing nothing"
ACTION_CLEAN = "cleaning"

# For nicer narration in the timestep log
ACTION_TO_PHRASE = {
    ACTION_COME_IN: "coming in",
    ACTION_LEAVE: "leaving the classroom",
    ACTION_SIT: "sitting down",
    ACTION_HAND_IN: "handing in their homework paper",
    ACTION_NOTHING: "doing nothing",
    ACTION_CLEAN: "cleaning the classroom",
}


def plural(n: int, singular: str, plural_form: Optional[str] = None) -> str:
    if n == 1:
        return singular
    return plural_form if plural_form is not None else singular + "s"


def format_group(n: int, gender: str, role: str, action: str) -> str:
    # Example: "1 female student coming in"
    # Example: "2 male students handing in their homework paper"
    role_word = role
    role_phrase = plural(n, role_word)
    return f"{n} {gender} {role_phrase} {ACTION_TO_PHRASE[action]}"


# -----------------------------
# Entity classes
# -----------------------------
@dataclass
class Entity(ABC):
    gender: str
    role: str
    timer: int = 0                  # starts with 0
    max_timer: int = 3              # maximum of 3 => max 4 time steps (0,1,2,3)
    status: str = ACTION_COME_IN    # last action / current action
    action_set: Tuple[str, ...] = field(default_factory=tuple)

    def get_id(self) -> str:
        return f"{self.gender}_{self.role}"

    @abstractmethod
    def get_action(self) -> str:
        """
        Return the action at the current time step.
        Protocol:
          - timer == 0 => "coming in"
          - timer == max_timer => "leaving"
          - otherwise choose from allowed actions.
        """
        raise NotImplementedError

    def step(self) -> str:
        """Get action, update status, increment timer."""
        action = self.get_action()
        self.status = action
        self.timer += 1
        return action

    def is_done(self) -> bool:
        """Remove from classroom after it has taken the 'leaving' action."""
        return self.status == ACTION_LEAVE


@dataclass
class Student(Entity):
    role: str = "student"
    handed_in: bool = False

    def __post_init__(self) -> None:
        if self.gender not in GENDERS:
            raise ValueError(f"Invalid gender: {self.gender}")
        self.action_set = (ACTION_SIT, ACTION_HAND_IN, ACTION_NOTHING)

    def get_action(self) -> str:
        if self.timer == 0:
            return ACTION_COME_IN
        if self.timer >= self.max_timer:
            return ACTION_LEAVE

        # Intermediate step: can hand in at most once.
        choices = list(self.action_set)
        if self.handed_in and ACTION_HAND_IN in choices:
            choices.remove(ACTION_HAND_IN)

        action = random.choice(choices)
        if action == ACTION_HAND_IN:
            self.handed_in = True
        return action


@dataclass
class Teacher(Entity):
    role: str = "teacher"

    def __post_init__(self) -> None:
        if self.gender not in GENDERS:
            raise ValueError(f"Invalid gender: {self.gender}")
        self.action_set = (ACTION_SIT, ACTION_NOTHING)

    def get_action(self) -> str:
        if self.timer == 0:
            return ACTION_COME_IN
        if self.timer >= self.max_timer:
            return ACTION_LEAVE
        return random.choice(self.action_set)


@dataclass
class Cleaner(Entity):
    role: str = "cleaner"

    def __post_init__(self) -> None:
        if self.gender not in GENDERS:
            raise ValueError(f"Invalid gender: {self.gender}")
        self.action_set = (ACTION_SIT, ACTION_NOTHING, ACTION_CLEAN)

    def get_action(self) -> str:
        if self.timer == 0:
            return ACTION_COME_IN
        if self.timer >= self.max_timer:
            return ACTION_LEAVE
        return random.choice(self.action_set)


@dataclass
class Dog(Entity):
    role: str = "dog"

    def __post_init__(self) -> None:
        # Keeping gender for dog too, since you requested it.
        if self.gender not in GENDERS:
            raise ValueError(f"Invalid gender: {self.gender}")
        self.action_set = (ACTION_SIT, ACTION_NOTHING)

    def get_action(self) -> str:
        if self.timer == 0:
            return ACTION_COME_IN
        if self.timer >= self.max_timer:
            return ACTION_LEAVE
        return random.choice(self.action_set)


# -----------------------------
# Classroom manager
# -----------------------------
class Classroom:
    def __init__(
        self,
        distractor_role: str = None,
        distractor_gender: str = None,        
        seed: Optional[int] = None,
        spawn_steps: int = 10,
        students_per_step: Tuple[int, int] = (1, 2),
        student_max_timer: int = 3,
        distractor_duration: int = 1,  # shorter than students (per your "only last ..." line)
        
    ) -> None:
        if seed is not None:
            random.seed(seed)

        self.spawn_steps = spawn_steps
        self.students_per_step = students_per_step
        self.student_max_timer = student_max_timer
        self.distractor_duration = max(0, distractor_duration)

        self.entities: List[Entity] = []

        # Stats
        self.summary: Dict[str, Dict[str, int] | int | List[int]] = {
            "students": {"female": 0, "male": 0},
            "teacher": {"female": 0, "male": 0},
            "cleaner": {"female": 0, "male": 0},
            "dog": {"female": 0, "male": 0},
            "homework": {"female": 0, "male": 0},
            "total time": 0,
            "unexpected event": [],  # timestamps while unexpected event happening
        }

        # Schedule distractor in the middle of the "active" phase
        if distractor_role is None:
            self.distractor_role: str = random.choice(["teacher", "cleaner", "dog"])
        else:
            self.distractor_role = distractor_role
        if distractor_gender is None:
            self.distractor_gender: str = random.choice(GENDERS)
        else:
            self.distractor_gender = distractor_gender

        self.distractor_start: int = max(1, spawn_steps // 2)
        # We mark timestamps as "unexpected event happening" across these steps:
        self.distractor_timestamps = list(
            range(self.distractor_start, self.distractor_start + max(1, self.distractor_duration))
        )
        self.distractor_spawned = False

    def _spawn_students(self) -> None:
        n = random.randint(self.students_per_step[0], self.students_per_step[1])
        for _ in range(n):
            g = random.choice(GENDERS)
            s = Student(gender=g, max_timer=self.student_max_timer)
            self.entities.append(s)
            self.summary["students"][g] += 1  # type: ignore[index]

    def _spawn_distractor_if_needed(self, t: int) -> None:
        if self.distractor_spawned:
            return
        if t != self.distractor_start:
            return

        max_timer = min(self.student_max_timer, self.distractor_duration)
        if self.distractor_role == "teacher":
            e: Entity = Teacher(gender=self.distractor_gender, max_timer=max_timer)
        elif self.distractor_role == "cleaner":
            e = Cleaner(gender=self.distractor_gender, max_timer=max_timer)
        else:
            e = Dog(gender=self.distractor_gender, max_timer=max_timer)

        self.entities.append(e)
        self.summary[self.distractor_role][self.distractor_gender] += 1  # type: ignore[index]
        self.distractor_spawned = True

    def _collect_actions(self) -> List[Tuple[Entity, str]]:
        actions: List[Tuple[Entity, str]] = []
        for e in self.entities:
            act = e.step()
            actions.append((e, act))
        return actions

    def _update_homework_stats(self, actions: List[Tuple[Entity, str]]) -> None:
        for e, act in actions:
            if isinstance(e, Student) and act == ACTION_HAND_IN:
                self.summary["homework"][e.gender] += 1  # type: ignore[index]

    def _timestep_statement(self, t: int, actions: List[Tuple[Entity, str]]) -> str:
        # Group by (gender, role, action)
        counts: Dict[Tuple[str, str, str], int] = {}
        for e, act in actions:
            key = (e.gender, e.role, act)
            counts[key] = counts.get(key, 0) + 1

        parts: List[str] = []
        # A stable-ish order: action first, then student/teacher/cleaner/dog; and by gender
        role_order = {"student": 0, "teacher": 1, "cleaner": 2, "dog": 3}
        action_order = {
            ACTION_COME_IN: 0,
            ACTION_HAND_IN: 1,
            ACTION_SIT: 2,
            ACTION_NOTHING: 3,
            ACTION_LEAVE: 4,
        }
        
        for (gender, role, act), n in sorted(
            counts.items(),
            key=lambda kv: (action_order.get(kv[0][2], 99), role_order.get(kv[0][1], 99), kv[0][0]),
        ):
            parts.append(format_group(n, gender, role, act))

        if not parts:
            return f"[timestamp {t}] (empty classroom)"

        return f"[timestamp {t}] " + "; ".join(parts) + "."

    def _prune_entities(self) -> None:
        self.entities = [e for e in self.entities if not e.is_done()]

    def run(self, print_log: bool = True) -> Dict:
        """
        Runs the simulation.

        - For t in [0, spawn_steps-1], we spawn 1-2 new students each step.
        - A distractor enters at distractor_start for a short duration.
        - After spawn_steps, stop spawning and let everyone leave naturally.
        - Ensure classroom ends empty.
        """
        t = 0
        logs: List[str] = []

        # Active phase
        for t in range(self.spawn_steps):
            self._spawn_students()
            self._spawn_distractor_if_needed(t)

            actions = self._collect_actions()
            self._update_homework_stats(actions)

            line = self._timestep_statement(t, actions)
            logs.append(line)
            self._prune_entities()

        # Cooldown phase: no new students; just let everyone leave
        while self.entities:
            t += 1
            actions = self._collect_actions()
            self._update_homework_stats(actions)

            line = self._timestep_statement(t, actions)
            logs.append(line)
            self._prune_entities()

            # Safety guard (should never hit)
            if t > self.spawn_steps + self.student_max_timer + 20:
                raise RuntimeError("Simulation did not terminate as expected.")
        self.summary["unexpected event"] = self.distractor_timestamps
        self.summary["total time"] = t + 1  # type: ignore[index]

        if print_log:
            for line in logs:
                print(line)

            print("\n=== SUMMARY ===")
            print(json.dumps(self.summary, indent=2))

        return logs, self.summary 


def main() -> None:
    # Tweak these knobs as you like
    dis_roles = ["teacher", "cleaner", "dog"]
    n_samples_per_role = 10
    length_range = list(range(6, 8))
    samples_json = []
    activity_str = set()
    for lenth in length_range:
        for role in dis_roles:
            for sex in GENDERS:
                for _ in range(n_samples_per_role):
                    now = time.time()
                    microsecond = int((now - int(now)) * 1000000)
                    classroom = Classroom(
                        seed=microsecond,
                        distractor_role= role,
                        distractor_gender= sex, 
                        spawn_steps=lenth,            # how long we keep introducing new students
                        students_per_step=(1, 2),  # randomly 1 or 2 students each step
                        student_max_timer=3,       # timer 0..3 (max 4 time steps per entity)
                        distractor_duration=2,     # short unexpected event
                    )
                    log, summary = classroom.run(print_log=True)
                    if not ("\n".join(log) in activity_str):
                        activity_str.add("\n".join(log))
                        for gen_of_focus in GENDERS:
                            summary['prompt_template'] = "Introduction: You will be given a sequence of activities in a classroom. Count {c1}\nActivities in the classroom: {c2};\n\nYour answer: there are/is "
                            summary['activities'] = "\n".join(log)
                            summary['question_1'] = f'how many {gen_of_focus} students had submitted their homework papers.'
                            # summary['prompt_template'] += f'\nA reiteration of the question: how many {gen_of_focus} students had submitted their homework papers?\nYour answer: there are '
                            # summary['prompt_template'] += f'\nAnswer with a digit.'
                            summary['answer_1'] = summary['homework'][gen_of_focus]
                            summary['question_2'] = f'A follow-up question: according to the activities, was there a {role} in the classroom?\nAnswer with Yes or No.'
                             
                            summary['answer_2'] = 1 # Did see the distractor
                            samples_json.append(dict(summary)) # deep copy      

                            summary['prompt_template'] = "Introduction: You will be given a sequence of activities in a classroom. Count {c1}\nActivities in the classroom: {c2};\n\nYour answer: there are/is "
                            summary['activities'] = "\n".join(log)
                            summary['question_1'] = f'how many {gen_of_focus} students had submitted their homework papers.'
                            # summary['prompt_template'] += f'\nA reiteration of the question: how many {gen_of_focus} students had submitted their homework papers?\nYour answer: there are '
                            # summary['prompt_template'] += f'\nAnswer with a digit.'
                            summary['answer_1'] = summary['homework'][gen_of_focus]
                            non_target_role = random.choice([ite for ite in dis_roles if not ite==role])
                            summary['question_2'] = f'A follow-up question: according to the activities, was there a {non_target_role} in the classroom?\nAnswer with Yes or No.'
                            summary['answer_2'] = 0 # Did not see the distractor
                            samples_json.append(dict(summary))  # deep copy
                    time.sleep(11.0/1000.0)  
    os.makedirs('./_datasets/IAB/', exist_ok=True)
    print('=========An Exmaple of Generated Data=========')
    print(json.dumps([samples_json[0]], indent=4))
    with open('./_datasets/IAB/iab.json', "w", encoding="utf-8") as f:
        json.dump(samples_json, f, ensure_ascii=False, indent=2)
    print(len(samples_json))

if __name__ == "__main__":
    main()
