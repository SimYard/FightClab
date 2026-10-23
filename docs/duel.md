# First duel skills: jab and its defense

Research and plan for the first two-robot task: one T800 jabs, the other defends. Collected on 2026-10-21 from a
literature search, the K-1 coach, the humanoid RL reviewer and a read of mjlab's code. The skill order comes from EA's
UFC 4 move list (bottom).

## Prior work

| Work | Setup | What we take |
|---|---|---|
| [Won, Gopinath, Hodgins 2021](https://ai.meta.com/research/publications/control-strategies-for-physically-simulated-characters-performing-two-player-competitive-sports/) | simulated boxing and fencing; motion prior (mixture of experts), then both players train a task encoder | Without the prior: wide stance, leaning back. Reward force dealt minus force received, clipped; penalize "stuck" (clinch, ropes). Punches emerged first, then blocking. |
| [Bansal et al. 2017](https://arxiv.org/abs/1710.03748) | kick-and-defend, sumo | Without an annealed dense bonus "the defender optimizes for being able to stand up but doesn't learn to defend". Sample opponents from the newer half of the history, not only the latest. |
| [Gleave et al. 2020](https://arxiv.org/abs/1905.10615) | attacks on frozen policies | Against a frozen opponent a policy can win by never standing up. Fine-tuning against one attacker forgets the rest. |
| [Haarnoja et al. 2024](https://arxiv.org/abs/2304.13653) | OP3 soccer | Skills first (incl. get-up), then self-play against early snapshots plus an untrained agent; the critic knows the opponent. |
| [SMPLOlympics 2024](https://arxiv.org/abs/2407.00187) | humanoid boxing | Two policies alternate, one frozen; no pool. Qualitative. |
| [Neural Categorical Priors 2023](https://arxiv.org/abs/2308.07200) | boxing over a discrete latent prior | Prioritized fictitious self-play: opponents weighted (1-p)^2 by the learner's win rate p; Elo over snapshots to catch cycling. |
| [Spherical Latent Motion Prior 2026](https://arxiv.org/abs/2603.01294) | boxing, two policies swap every 250 epochs | Dense shaping made them stand close without striking; contact-only rewards made them punch repetitively. 8 h on 2x RTX 4090. |
| [RoboStriker 2026](https://arxiv.org/abs/2608.16195) | two Unitree G1s, Isaac Lab, one RTX 4090, 4096 envs | Tracker, 32-d latent, warm-up against a standing opponent, then neural fictitious self-play. Failures: no warm-up gives ~0; a static-target specialist fails against a moving one; naive self-play cycles; joint-level self-play drops the guard. |
| [MAAIP](https://arxiv.org/abs/2311.02502) | two boxers, one shared policy with an agent id | Parameter sharing works for symmetric fighters. |

Hardware: Unitree's "Iron Fist King" bouts were [teleoperated](https://www.heise.de/en/news/Iron-Fist-Unitree-plans-boxing-match-between-humanoid-robots-10348477.html);
nobody has shown a learned block on a real humanoid. The closest is [dodgeball on a G1](https://arxiv.org/abs/2607.28623), 95% dodged.

Reaction time: the [Ishikawa lab hand](https://spectrum.ieee.org/robot-hand-beats-you-at-rock-paper-scissors-100-of-the-time)
sees at 1000 fps and reacts in ~1 ms; a teleoperated humanoid punching measured ~70 ms ([arXiv 2409.04639](https://arxiv.org/abs/2409.04639)).
Feints slow even experienced boxers ([Loughborough](https://repository.lboro.ac.uk/articles/journal_contribution/Timing_a_fake_punch_Inhibitory_effects_in_a_boxing-specific_spatial_attention_task/23925465)).

## The K-1 coach

Jab, standing first (thresholds are starting values to calibrate; H = 1.75 m):
- starts and ends in guard (hands within 0.14 H of the head, elbows within 0.13 H);
- no wind-up (pull-back 0.02 H or less; the hand moves before the pelvis turns);
- straight path (within 0.05 H of the line), reach 0.90 or more of the arm, elbow bent 5-20 deg;
- peak fist speed 2.5 H/s or more (4.4 m/s), guard to extension in 0.25 s or less, at contact still 80% of peak;
- rear glove within 0.14 H of the head throughout; back in guard within 0.25 s; feet slide 0.02 H or less.

Defense: the high guard alone probably fails (a jab splits it down the centre line); test with a statue first. Train
the **rear-hand catch** first: the rear glove moves 0.05-0.10 H forward and in and takes the fist in front of the face.
- the fist touches the defender's rear hand in 85% or more of head jabs, never the head or torso at speed;
- glove onset between jab onset and 0.12 s after, in place 0.03 s before contact; travel 0.10 H or less, not past the
  head midplane by more than 0.03 H; back in guard within 0.25 s;
- the lead hand stays home, no leaning back (10 deg or less), no retreat (pelvis 0.05 H or less);
- perception latency, or the catch becomes superhuman.

Attack variation, each step once the defender catches 80%: fixed-tempo head jabs, random tempo, random aim on the head,
distance 0.9-1.1 reach (step jab), double jab, body jabs (25-30%), feints last (only if the defender starts guessing).

Generated jab (`k1/strikes.py`), after two of the coach's reviews: the first version "hit with the forearm, a slap"
(elbow bent 44 deg at contact, upper arm below horizontal, fist on its side, hips still), the second missed 11 cm to
the right and gave the jab away by turning the shoulder 0.3 s early. Its reference fist stops at full extension, so
**the target surface in the duel sits 2-3 cm short of it**: the arm is then bent 34-43 deg and the fist moves at
1.4-1.7 m/s (2 m/s in the step jab, which lands the fist before the foot with the body still moving). Further short
the arm is bent like the slap again (60 deg at 8-10 cm). So the clean-hit speed threshold for the jab goes under
~1.3 m/s, or the reference jab can never score.

Tracking it (`Fightclab-K1-Footwork-Tracking-T800`, from the footwork policy): with mjlab's tracking alone the robot
kept its forearm up by its head, the fist 13 cm short: averaged over 14 bodies, one fist's error costs almost nothing
(the kick's foot had the same problem, hence `feet_height`). A reward on the fists alone (`bodies_pos`, weight 3,
std 0.1 m) got the jab in about 1500 iterations: reach 0.52 m of the reference's 0.55, fist 3.0 m/s of 3.0-3.2.
**Every strike gets a reward on its striking body** (the shin for a low kick, the knee for a knee).

Stress (`tools/stress.py`, the footwork-and-jab policy fw_track_v16, 64 envs x 10 s): no falls with the floor at
friction 0.3 or 1.2, with shoves of 0.3 or 0.6 m/s, or with every motor at 80% or 60% of its torque (error 1.4-1.8 cm);
at 40% it falls 12 times a minute. The damage model's first anchor: 40% of torque lost over the whole body is
survivable, 60% is a knockdown. Next: per limb (one knee, the legs, the arms), as hits land.

## The RL review

- Train the jab in the duel scene against a T800 "statue" holding its guard (and moving a little), not in an empty one.
- Jabs from video are slow: fist peaks at 2.3-2.6 m/s in our clips (a human jab is 6-9 m/s). Set the clean-hit speed
  threshold from the clip, and the speed by time-warping it.
- The footwork tracking rewards punish defense: the adaptive tracking widths collapsed (moving a wrist 15 cm loses 90%
  of the body reward). The defender tracks legs and torso only, at fixed widths; arm penalties are gated off during jabs.
- Defender sees the attacker's fists, elbows, head and torso (positions, fist velocities) in its own heading frame, 3-5
  frames, with noise and 20-140 ms delay. The attacker's phase and target go to the critic only.
- "Do nothing" control: if a statue is hit in under ~30% of jabs, change the attack (body jabs, second punch), not the
  defender's reward. A perfect blocker must beat the statue's return by 10% or more.
- Frozen opponent inside the env: TorchScript or ONNX, its observations rebuilt exactly as trained, a zero-dim action
  term; known-answer test: alone in the duel env it must still track with 1 cm error.
- Rounds: switch at 70% wins or a 300-iteration plateau; pool of past snapshots in blocks of envs; log a cross-play
  matrix, statue hit rate, false alarms, reaction latency, guard-open time, fist speed at impact.

## What mjlab gives and what we write

| Need | mjlab | Work |
|---|---|---|
| two T800s in a scene | `SceneCfg.entities` is a dict; per-entity offset via `init_state` | add `"opponent"`, 1.2 m away, turned 180 deg |
| actions for the second robot | `ActionTermCfg.entity_name` | an action term with `action_dim` 0 that computes its own targets |
| frozen policy in the env | not supported | load TorchScript/ONNX in that term (pattern: `LearnedActuator`) |
| fist hits head or glove | `ContactMatch` per entity, `found`/`force`/`pos` | one sensor per target (head, torso, gloves) |
| reference for the opponent | `MotionCommand(entity_name=...)` | per-entity offset and yaw (both references sit on one spot now) |
| contacts for two robots | `nconmax` 35, `njmax` 250 per world | raise both |
| hard-coded `"robot"` | events, IMU sensors, self-collision, terminations | copy for the opponent |

## Approach

Training a jab policy, then a defender policy against it at the joint level, then alternating, is the plan the
literature warns about: defenders that lean back or stand still, guards that drop in self-play, cycling. Instead, one
body with many skills and a small brain on top:

1. **Skills are references, generated from video plus IK.** Like `k1/steps.py` for footwork: the jab clip (Varga, in
   profile) gives the shape; IK moves the fist's end point onto any target (head, chin, body, with offsets) and
   time-warping sets the speed. The catch is generated the same way: the rear glove goes to an intercept point. Every
   clip starts and ends in the same guard (`GUARD_POSE`), so skills chain without blending. One clip gives unlimited
   target-conditioned variants.
2. **One tracker for all skills.** The footwork tracker grows into a library tracker (stance, steps, jab, catch, later
   cross, low kick, get-up), trained with a statue partner so fists landing on arms don't break tracking, and with
   pushes on the arms and head. A new skill is new clips and a fine-tune, not a new policy.
3. **A small duel policy picks skills, as in UFC's move list.** At ~10 Hz it outputs a skill (jab, catch, step,
   stay), a target and a timing; the tracker executes. The guard and stance cannot drop because the tracker holds them.
   The defender's problem becomes when and where to catch, which is exactly the skill, and a small action space learns
   in minutes.
4. **Both robots share that policy** (each from its own view), so every duel env gives two samples and rsl_rl stays
   single-agent (two entries per env). In part of the envs one robot is a frozen past snapshot or a scripted bot
   (statue, random jabber): no cycling or forgetting. Zero-sum reward: clean hits dealt minus received, small shaping
   for range and facing, annealed.
5. **Latency by teacher and student.** The duel policy trains on exact state; a student with 20-140 ms delay and noise
   is distilled from it (DAgger).

Order: jab and catch generators, then the library tracker with a statue, then a scripted jabber against a learning
catcher (warm-up), then shared self-play with a pool. Later the skill commands can become a learned latent space
(PULSE, ASE) for blends and combinations.

## Data

Hands and feet must both be in frame: a jab's step and weight shift come from the legs, and GVHMR invents legs it
cannot see. Keep only pieces where ViTPose sees both ankles and both wrists with confidence 0.5 or more in 95% of
frames (`preprocess/vitpose.pt`, keypoints 9, 10, 15, 16).

- Jab: "Combining Footwork & Punches" (Gabriel Varga), 8:30-11:40, in profile, about 10 jabs at ~2 m/s
  (`~/fw_clips/jab_vg_01..04`). 8:30-9:30 (`jab_vg_01`) passes: ankles seen in 95% of frames, wrists in 100%. The
  next three minutes lose the ankles in 11-16% of frames.
- Defense: "How To Build Pro Defense Skills" (Varga and a partner, static camera, both in profile): two people, so it
  needs GVHMR run per person, and only the wide shots (the close-ups cut the legs off). "How to block and parry, even
  against a powerful jab" (handheld, outdoors) and "K1 Kickboxing 5 Hard Hitting Counters to the Jab" (Mick Crossland)
  as backups.
- Low kick: "Right Low Kick by Ilias Ennahachi".

## Skill order (from EA's UFC 4 move list)

| Tier | Hard part | Moves | Status |
|---|---|---|---|
| 0 movement | | stance, steps, switch stance, head movement | stance and steps on T800 |
| 1 hands, feet planted | arms and torso turn | jab, cross, hooks, uppercuts (head and body); high and low block | next: jab and catch |
| 2 hands with the body | big turn, weight shift | overhands, elbows | |
| 3 low kicks | short single-leg support | lead and rear leg kicks, front kick (teep), side and oblique kicks to the leg; leg catch | |
| 4 mid and high kicks | long single-leg support, hip range | body and head kicks, switch kick, side kicks, knees, question mark, axe | |
| 5 spinning | turn on one leg, blind strike | spinning backfist and elbow, spinning heel kick, spin side kick | |
| 6 airborne | both feet off the floor | superman punch, flying knee, jumping switch and roundhouse kicks, tornado | helicopter kick on 7 robots |
| 7 with an opponent | contact | clinch knees and elbows, takedown defense (ground work out of scope) | |

Not in UFC's list but needed by our rules: get-up after a knockdown (knockout = not up within N seconds), before the
first bouts. Feints in UFC are a cancelled strike (strike, then block), not a move of their own; ours come from a jab
clip cut short.

Sources for the list: [EA's UFC 4 manual](https://ea.com/able/resources/ufc/ufc-4/xbox-one/manual),
[Outsider Gaming's UFC 4 striking guide](https://outsidergaming.com/ufc-4-complete-striking-guide-tips-and-tricks-for-stand-up-fighting/).

## Open questions

- Main fighter: CLAUDE.md still names G1 and `Fightclab-Duel-G1`; all the K-1 work is on T800.
- The clean-hit threshold on robots with video-speed clips: calibrate on statue hits.
