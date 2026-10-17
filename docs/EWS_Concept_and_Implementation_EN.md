# An Early Warning System for Embryo Culture Conditions

### Concept, design decisions and implementation in our laboratory

*A description for clinicians, laboratory directors and department heads. Technical
machinery is explained in plain language; no prior knowledge of statistics is assumed.*

---

## Table of contents

1. [What this document is](#1-what-this-document-is)
2. [Why a laboratory needs an early warning system](#2-why-a-laboratory-needs-an-early-warning-system)
3. [What "early" actually means: the time budget of a cycle](#3-what-early-actually-means-the-time-budget-of-a-cycle)
4. [Design approaches to this problem, and the one we took](#4-design-approaches-to-this-problem-and-the-one-we-took)
5. [How our system works](#5-how-our-system-works)
6. [The system inside laboratory quality control](#6-the-system-inside-laboratory-quality-control)
7. [The role of the AI component — and its boundaries](#7-the-role-of-the-ai-component--and-its-boundaries)
8. [What the implementation measured](#8-what-the-implementation-measured)
9. [What the system deliberately does not do](#9-what-the-system-deliberately-does-not-do)
10. [Deployment, portability and data protection](#10-deployment-portability-and-data-protection)
11. [Current status and what comes next](#11-current-status-and-what-comes-next)
12. [Glossary](#12-glossary)

---

## 1. What this document is

This is a description of a working system that we designed, built and calibrated on our
own laboratory data: a daily monitor that compares what the laboratory actually produced
against what it should have been expected to produce, and raises a flag when the two stop
agreeing.

The document explains three things:

- **why** such a system is needed at all in an IVF laboratory, in terms of the failures it
  is meant to catch;
- **how ours is built**, including the decisions that shaped it and the reasoning behind
  each one;
- **how it is used in day-to-day quality control**, which is the point of the whole
  exercise.

It is not a research paper, and it does not attempt to rank our system against published
ones. Where other design approaches are described, they are described to explain why we
chose the design we chose — not to claim that ours performs better.

One framing note that runs through everything below. The system produces **observations and
reasons to look into something**. It does not produce diagnoses, does not issue
instructions, and does not evaluate any individual member of staff. Every output separates
what was measured from what is merely a hypothesis about the cause.

---

## 2. Why a laboratory needs an early warning system

### 2.1 The laboratory is a life-support system, and the embryos are its only sensor

An IVF laboratory is a small, tightly controlled environment sustaining living cells for
five to six days. Temperature, pH, osmolality, gas phase, air quality, the composition of
each media lot, the plastics, the oil, the timing and the handling technique all feed into
one output: how many oocytes become zygotes, how many zygotes become blastocysts, and how
good those blastocysts are.

Every one of those inputs is monitored directly in a well-run laboratory — incubators log
temperature and gas, media come with certificates, air is filtered and tested. But direct
monitoring has a blind spot that matters: **each parameter can sit inside its own
specification while the combination is still wrong for the embryos.** A media lot can meet
every release criterion and still underperform. An incubator can hold its setpoint and
still expose dishes to more recovery time than usual because workload changed. Air quality
can pass its scheduled test and drift between tests.

The embryos integrate all of it. They are the only sensor that measures the thing we
actually care about — and they are a sensor we already read every day, cycle by cycle, in
the routine laboratory record.

### 2.2 The failures that matter are silent and slow

The failures that damage an IVF programme are rarely dramatic. A power outage or an alarm
condition announces itself; someone acts within minutes. The dangerous failures are the
ones that do not announce themselves:

- a media lot that is subtly worse than the previous one;
- a gas mixture drifting a few tenths of a percent;
- a change in workload that lengthens the time dishes spend out of the incubator;
- a new consumable that is embryotoxic at a level below any release specification;
- a slow change in a technique — the timing of a fertilisation check, the handling during
  denudation — that no one decided to make and no one noticed;
- a slow deterioration of an incubator's actual performance versus its logged performance.

None of these trip an alarm. All of them show up in the biology, and only in the biology.
The question is how long it takes to notice.

### 2.3 The conventional feedback loop is far too slow

The traditional answer is the monthly or quarterly KPI review: the laboratory computes
fertilisation rate, blastocyst development rate, good-blastocyst rate and so on, compares
them against consensus thresholds, and discusses them at a quality meeting.

This is necessary and we have not removed it. But as an *early warning* mechanism it has
three structural problems.

**It is retrospective by construction.** A monthly review held in the first week of
February looks at January. If something changed on 5 January, the laboratory has already
run a month of affected cycles before anyone looks, plus however long the review takes to
convene, plus however long it takes to decide the change was real rather than noise. Two
to three months of cycles is a realistic exposure.

**It waits for the wrong signal if it waits for pregnancy.** Clinical pregnancy is the
outcome everyone cares about, and it is the slowest signal in the building. From oocyte
retrieval, a fresh transfer plus a pregnancy test is six to eight weeks; with freeze-all
policies and deferred transfers it is longer. By the time a pregnancy rate has moved
enough to be visible in a small laboratory's numbers, the causal event is a season old.

**It has almost no statistical power at the volume of a normal clinic.** This is the point
that is most often underestimated. Our laboratory runs roughly **10 oocyte retrievals and
about 70 fertilised oocytes per month**. A month-to-month comparison of blastocyst rate on
ten cycles is essentially unreadable — the natural variation between patients swamps any
process change short of a catastrophe. Looking at the same numbers monthly, in isolation,
produces a reliable stream of both false alarms and missed events.

### 2.4 The two errors, and why the false alarm is the more expensive one

Any monitoring system makes two kinds of mistake: it misses a real event, or it flags
something that was not there.

Missing an event is obviously costly — cycles are lost, and in IVF a lost cycle is not
recoverable.

But a false alarm has a cost that compounds. An alarm in this context means a real
investigation: pulling incubator logs, checking media lots, reviewing air quality records,
reviewing handling, possibly quarantining a lot and possibly pausing part of the workflow.
That is hours of senior staff time and a disruption to normal operations. And there is a
second-order effect that is worse than the first: **after two or three empty
investigations, people stop reading the system.** A monitor that is not trusted is
functionally identical to a monitor that does not exist.

This asymmetry is not a side note in our design — it is the constraint the whole system is
built around, and it drives the decision described in section 6.4.

### 2.5 The confounder that ruins naive monitoring: the patients change too

The final reason a laboratory needs a purpose-built system rather than a simple chart is
that laboratory output is not only a property of the laboratory.

A run of poor-prognosis patients — older, lower ovarian reserve, more repeat attempts,
poorer sperm — produces fewer blastocysts from the same laboratory doing the same work.
A run of good-prognosis patients hides a real deterioration. At ten cycles a month, a
handful of unusual patients is enough to move any monthly rate by several percentage
points.

A monitoring system that ignores this does one of two things: it raises alarms about
patient mix and calls them laboratory problems, or it is set so insensitively that it never
raises anything. Neither is useful. **Separating "the process changed" from "the patients
changed" is the central technical problem**, and it is what most of our design effort went
into.

---

## 3. What "early" actually means: the time budget of a cycle

It is worth being explicit about the timeline, because every design decision follows from
it.

| Moment | What becomes known | Available as a signal? |
|---|---|---|
| Day 0 — oocyte retrieval | Patient characteristics, follicle count, oocytes collected, mature oocytes, sperm parameters, who performed what | Yes, immediately |
| Day 1 — fertilisation check | Normally fertilised, abnormally fertilised, degenerated | Yes, next day |
| Day 5–6 — cycle closure in the laboratory | Cleavage, blastocysts, good-quality blastocysts, vitrified | Yes, within a week |
| Weeks later | Transfer performed | Depends on policy |
| 6–8+ weeks | Pregnancy test, clinical pregnancy | Very late |
| ~9 months | Live birth | Far too late for process control |

The design question is where to place the evaluation point. Two answers were rejected and
one was chosen.

**Rejected: evaluate at day 0 or day 1.** It is technically possible to raise a flag as
soon as fertilisation numbers arrive. We decided against it, as a fixed rule. A single
cycle's fertilisation result is a very noisy quantity, and an alarm raised before the
laboratory picture is complete is a false alarm far more often than not. Given the cost
structure in section 2.4, premature alarms destroy the system's credibility faster than a
missed signal does.

**Rejected: wait for pregnancy.** This is the outcome that matters, but it arrives too late
to be a control signal, and it is contaminated by everything outside the laboratory — the
transfer itself, the endometrium, the clinical protocol. We do use it, but as a slow
confirming loop rather than as a trigger (section 6.8).

**Chosen: evaluate once, at day 6, when the laboratory part of the cycle closes.** This is
the earliest moment at which the full laboratory funnel is visible, and it is roughly six
weeks earlier than any pregnancy-based signal. Day 0 and day 1 remain in the system as
**registration** points — they record what became known when it became known, and they
freeze the forecast (section 5.3) — but they evaluate nothing and raise nothing.

---

## 4. Design approaches to this problem, and the one we took

Several families of design have been proposed for putting a predictive model into
laboratory monitoring. They differ in what the model predicts, what data they require, and
what they are therefore able to detect. Understanding those differences is the fastest way
to explain our own design, so they are summarised here — as engineering alternatives, not
as competitors.

### 4.1 Approach A — an image-based deep-learning KPI from time-lapse

**The idea.** A neural network looks at embryo images from a time-lapse incubator at a
fixed hour of development and predicts, for each embryo, its developmental fate or its
implantation potential. The average predicted value across a batch of embryos becomes a
new KPI — one that is available on day 3 rather than in eight weeks, and that tracks
clinical outcome far more closely than classical day-2/day-3 morphology grades do. That KPI
is then plotted on a control chart, and shifts in it are treated as early warnings about
culture conditions or about individual operator performance.

**What it requires.** A time-lapse system on every dish, a trained network, and enough
throughput to keep the batch-level KPI stable.

**What it detects well.** Anything that alters embryo morphokinetics — which is a very wide
class of culture-condition problems, including media lots — and it detects it at the level
of the individual embryo, which is where the statistical power is.

### 4.2 Approach B — AI-based quality assurance of staff performance

**The idea.** The same kind of predictive model is used differently. Instead of tracking a
KPI over time, the model's predicted outcome for a set of embryos is compared against the
actual outcome achieved after a specific procedure performed by a specific person —
transfer, vitrification, warming, biopsy. If a provider's actual results sit consistently
below what the model predicted for those same embryos, that discrepancy is the signal. The
model, in effect, supplies the control group that a small clinic can never assemble.

**What it requires.** A model whose predictions are credible at the level of the individual
embryo, plus enough procedures per person to compare.

**What it detects well.** Technique problems that are invisible in aggregate KPIs because
they are diluted across the whole team.

**The framing we adopted from it.** The useful idea here is not the staff ranking — it is
the *prediction as the reference*. Comparing what happened against what was expected for
this particular material is a far cleaner question than comparing this month against last
month.

### 4.3 Approach C — fusion of clinical data and images

**The idea.** Build two models — one on the patient's clinical data, one on embryo images —
and combine them. The consistent finding of this line of work is instructive: the clinical
model carries a large share of the predictive signal on its own, the image model carries
less on its own, and the combination performs better than either. Patient age and related
clinical factors dominate; embryo appearance adds information on top of them.

**What we took from it.** Two things. First, that a model built on ordinary clinical and
laboratory data — the data every clinic already records — is a legitimate and substantial
predictor, not a fallback. Second, and more importantly for us: **the patient's clinical
profile is the dominant source of variation in the outcome.** For a monitoring system, that
is the same statement as "the patient mix is the main confounder", which is exactly the
problem set out in section 2.5.

### 4.4 What we could not use, and what we did instead

We do not have time-lapse incubation. The image-based half of approaches A and C is
therefore unavailable to us, and no amount of engineering recovers it.

What we did was to keep the *construct* and change the *sensor*:

- from approach A we kept the core construct — **do not wait for the outcome; monitor a
  leading indicator on a proper control chart**, and keep the culture-conditions loop
  separate from the operator loop, because they have different causes and different
  remedies;
- from approach B we kept **prediction-as-reference**: the comparison is against what was
  expected for this specific material, not against a calendar period;
- from approach C we kept the finding that **clinical data alone carries a large share of
  the signal**, which is what makes our substitute sensor viable at all.

In place of a per-embryo image prediction, our system uses a **per-cycle predicted yield
envelope**: before culture begins, a set of models predicts how many zygotes, blastocysts,
good blastocysts and vitrified embryos this particular cycle should produce, and expresses
that prediction as a range rather than a single number. The laboratory's actual result is
then compared against that frozen range.

### 4.5 Design comparison

| | Image-based KPI (A) | AI-QA of staff (B) | Clinical + image fusion (C) | Our system |
|---|---|---|---|---|
| What the model predicts | Fate / implantation of each embryo, from images | Implantation of each embryo, from images | Pregnancy, from clinical data + images | Expected count at each funnel stage for this cycle, from clinical data |
| Input required | Time-lapse on every dish | Time-lapse plus per-procedure records | Clinical record plus blastocyst images | Routine oocyte-retrieval record only |
| Unit of observation | Embryo | Embryo, grouped by provider | Embryo / transfer | Embryo for rates; cycle for the envelope |
| Primary use | Leading KPI on a control chart | Retrospective comparison of provider vs prediction | Embryo selection and outcome prediction | Continuous process control across three loops |
| How patient mix is handled | Implicitly — the images already reflect the material | Implicitly, same | Explicitly — clinical features are model inputs | Explicitly — the expected range is computed per cycle from that patient's data |
| Absolute reference | Clinic's own outcome threshold | The model's prediction | Not applicable | Consensus competency and benchmark values, held separately |
| Prerequisite we lacked | Time-lapse | Time-lapse | Time-lapse | — |

The last row is the reason for the design, stated plainly. The rest of the table is meant
to make the trade-offs visible: our design gives up embryo-level morphological information
and gains applicability to a laboratory that records nothing beyond its normal
oocyte-retrieval table.

---

## 5. How our system works

### 5.1 Three reference frames, none of which replaces the others

The single most important structural decision is that the system judges every result
against **three independent references** at once, and reports all three.

| Reference frame | What it is | The question it answers | Does it move with the laboratory? |
|---|---|---|---|
| **Relative** | The laboratory's own declared reference period | "Have we changed compared with ourselves?" | Yes |
| **Absolute** | Published consensus competency and benchmark values | "Where do we sit against the standard?" | No, but identical for every patient |
| **Expected-for-this-patient** | A mechanistic cascade model that estimates the expected yield for this specific patient | "Where are we against what was reasonable for *these* patients?" | No, and it accounts for the patient |

Each one alone fails in a specific way, and the failures are complementary:

- The **relative** frame is self-referential. If a laboratory drifts slowly downwards and
  its reference period is periodically redeclared, the charts will see nothing at all —
  the new normal simply becomes normal.
- The **absolute** frame does not drift, which is its virtue, but it is a single number
  applied to every patient regardless of prognosis. It cannot distinguish a poor month from
  a poor cohort.
- The **expected-for-this-patient** frame does not drift *and* accounts for the patient,
  but it is a model, and models can be wrong or out of date.

Used together they interrogate each other. This is also what produces the system's most
useful output, which is not an alarm but a *verdict*:

| Relative frame | Absolute frame | Verdict issued |
|---|---|---|
| Drop detected | Still above consensus benchmark | **A trend, not a failure** — look at patient mix and stimulation, not at the culture protocol |
| Drop detected | Below competency | Review the laboratory protocol |
| Silent | Below competency | A long-standing difference in level — schedule a planned review, this is not an event |

The first row is the one that saves the most wasted effort. If a laboratory is performing
*above* the external standard, a deviation from its own history means "something is
different", not "something is wrong". Treating those two as the same thing is the fastest
way to generate the empty investigations described in section 2.4.

### 5.2 What the system measures

The system works from the laboratory's ordinary oocyte-retrieval record. From it, eight
rates are computed and assigned to one of three **loops** — groups of metrics that share a
plausible cause:

| Loop | Metric | Definition |
|---|---|---|
| **Physician** | OCCR | Oocytes collected ÷ follicles aspirated |
| | MIIR | Mature (MII) oocytes ÷ oocytes collected |
| **Embryologist / ICSI** | FR | Normally fertilised (2PN) ÷ mature oocytes |
| | DegR | Degenerated oocytes ÷ mature oocytes (upper control — lower is better) |
| | 3PNR | Abnormally fertilised ÷ oocytes collected (upper control) |
| **Culture conditions** | ClR | Cleaved ÷ 2PN |
| | BlR | Blastocysts ÷ 2PN |
| | TGBDR | Good-quality blastocysts ÷ 2PN |

Splitting them this way is what makes an alert actionable. A drop in blastocyst rate while
fertilisation rate is normal points at culture conditions — media, lot, incubator, gas, air.
A drop in fertilisation rate while oocyte maturity is normal points somewhere else
entirely. The system performs this attribution automatically, and always presents it as a
hypothesis rather than a conclusion.

Two further rules matter for anyone reading the numbers:

- A rate is always computed as *total numerator ÷ total denominator* across the group,
  never as the average of per-cycle rates. Averaging rates lets one two-oocyte cycle carry
  the same weight as one twenty-oocyte cycle.
- A zero is an observation. Only a genuinely empty field is a missing value. This sounds
  pedantic; in practice it is the difference between a real signal and a silently discarded
  one.

### 5.3 The frozen forecast envelope

Before culture begins — on the day of oocyte retrieval, when nothing about the outcome is
yet known — the system runs a set of predictive models on that cycle's clinical data and
records, for each stage of the funnel, a **range**: an optimistic value, a central value
and a pessimistic value (formally the 10th, 50th and 90th percentiles).

That range is then **frozen**. It is written once, stamped with the model version and the
run identifier, and never recomputed. This is a hard rule, and it is worth explaining why
it matters so much.

A forecast that can be revised after the answer is known is not a forecast. If the system
were allowed to re-run the model at closure, it would inevitably produce a range that fits
the observed result better, and the comparison would become meaningless — a bookmaker
rewriting the odds after the race. Freezing the envelope at day 0 is what makes the
day-6 comparison an actual test.

The frozen envelope then serves two distinct purposes:

1. **A per-cycle expectation check.** Did this cycle land inside its predicted range? A
   cycle that produces two blastocysts when the range was 1–4 is unremarkable. A cycle that
   produces two when the range was 6–11 is a data point.
2. **A model audit layer.** Across many cycles, roughly the expected share should land
   inside the range. If that share falls, either the patients have changed, or the model has
   drifted — and the system checks those in a fixed order before it will consider a
   laboratory explanation (section 6.6).

### 5.4 The three checkpoints

| Checkpoint | When | What happens | Can it raise an alarm? |
|---|---|---|---|
| **T0** | Day of oocyte retrieval | Register the cycle; compute and **freeze** the forecast envelope; record case-mix and the physician-loop counts | No |
| **T+1** | Fertilisation check | Record 2PN, 3PN, 1PN, 0PN and degenerated, each with its own date | No |
| **T+6** | Laboratory closure of the cycle | Compute all three loops, all residuals against the frozen envelope, update the control charts, gather context, issue a verdict | **Yes — the only evaluation point** |

T0 and T+1 raise nothing by design. They exist to make the record honest: to fix the
forecast before the answer is known, and to timestamp the fertilisation numbers separately
so that at closure it is possible to tell an ICSI problem from a culture problem. Without
separate dates, those two causes collapse into one number and the attribution in section 5.2
becomes impossible.

### 5.5 The control charts, in plain language

The core monitoring device is a **cumulative score**, one per monitored metric. The idea is
simple and does not require any statistics to understand:

> Every time an embryo does better than expected, the score goes down (and is not allowed
> below zero). Every time an embryo does worse than expected, the score goes up. As long as
> the laboratory is performing as usual, the score bumps around near zero. If performance
> genuinely shifts downwards, small deficits accumulate instead of cancelling out, and the
> score climbs steadily. When it crosses a defined line, the chart signals.

Three properties of this device are what make it suitable here.

**It accumulates.** It does not ask "was this month bad?" — a question that ten cycles
cannot answer. It asks "has the shortfall been piling up?", which many small observations
*can* answer.

**Its unit is the embryo, not the cycle.** At ~10 cycles per month there is nothing to
monitor; at ~70 fertilised oocytes per month there is. This is why the charts are built on
embryo-level outcomes.

**It remembers.** The score persists between daily runs. A chart that resets every morning
is not a sequential monitor; it is a series of unrelated snapshots.

There are two lines, not one:

- **watch** — the lower line. Something is worth looking at. It appears on the panel and
  wakes nobody. Its cost is a few minutes of attention.
- **alert** — the upper line. This initiates a real investigation: incubator, media, lot,
  air, handling. Its cost is hours of work and possible disruption.

The two-level structure is deliberately the same warning/action logic that is standard in
clinical laboratory quality control, so that it reads as familiar rather than novel.

### 5.6 One more refinement: not every metric may go on a chart

A metric qualifies for a sequential chart only if its **denominator is objective**.

The clearest case is OCCR — oocytes collected divided by follicles aspirated. The
denominator is the physician's count of follicles at retrieval, which is a judgement, and
judgements drift with practice. In our own data, OCCR fell by 7.6 percentage points
between two periods (0.797 → 0.721, p = 0.001), and the drop occurred within a single
physician rather than through a change of staff. But over the same interval, follicles
counted per cycle rose 12% (16.7 → 18.6) while oocytes actually collected per cycle did not
move at all (13.3 → 13.4, p = 0.84) and mature oocytes per cycle rose slightly (10.1 →
10.3).

The denominator changed; the result did not. A control chart on OCCR would have been
monitoring how the laboratory counts, not how it works. OCCR therefore stays in the
absolute-consensus reporting, where the level is shown honestly, but it gets no sequential
chart. The rule is enforced in configuration: a metric flagged as having a subjective
denominator cannot be placed on a chart.

For the physician loop, MIIR is used instead — its denominator is the number of oocytes in
the dish, which is an objective count.

---

## 6. The system inside laboratory quality control

This section is the operational core: what the system actually changes about how quality
control is done.

### 6.1 Where it sits

The system does not replace any existing element of the quality management system. Incubator
alarms, media certificates, air quality testing, equipment qualification, competency
assessment and the periodic consensus KPI review all continue exactly as before.

What it adds is a **continuous, automated layer between them**. Concretely:

| Existing QC element | What it monitors | Its blind spot | What the system adds |
|---|---|---|---|
| Incubator alarms and logs | Physical parameters against setpoints | A parameter can be in spec and still be wrong for the embryos | Reads the biological consequence directly |
| Media / consumable certificates | Release specification of a lot | A lot can pass release and still underperform in use | Tracks output by lot-use period |
| Periodic KPI review | Rates against consensus thresholds | Retrospective; no power at monthly volume | Sequential monitoring on embryo-level data, evaluated daily |
| Competency assessment | Individual performance at a point in time | A snapshot, and a scheduled one | Continuous monitoring of the process loop, not the person |

### 6.2 What runs, and how often

| Cadence | Process | Output |
|---|---|---|
| **Daily** | Synchronise new cycles; register and freeze envelopes for new retrievals; evaluate every cycle that closed; update the charts; gather context for anything that fired; rebuild the panel | Updated dashboard; any new alert cards |
| **Weekly** | Ingest the incubator log export; enrich already-closed cycles and already-open alerts with environmental context; recompute what counts as an unusual environmental load | Enriched alert cards; refreshed context baselines |
| **On demand** | Full retrospective replay over the entire history | "What would the system have said?" — used before go-live and after any change |
| **On trigger** | Threshold recalibration | New thresholds plus a calibration manifest |

The environmental enrichment is deliberately a **soft** step. Our incubator log export is
produced weekly, not streamed, so at the moment a cycle closes those logs may not exist yet.
The day-6 evaluation therefore must complete without them. A cycle whose logs have not
arrived is marked *pending* — explicitly not "no events occurred". When the logs do arrive,
the enrichment pass can *raise* an already-open watch to alert, but it can never create an
alert that did not exist and can never retroactively withdraw one. Silence about the
environment is never allowed to look like reassurance about the environment.

### 6.3 What a QC officer actually receives

**An alert card**, one per firing cycle, containing:

- the cycle identifier (pseudonymised), the closure date, the level (watch or alert), and
  the status of environmental enrichment;
- what fired: which metric, the observed rate, the denominator, the confidence interval,
  the consensus threshold and which side of it the value sits on; or, for envelope-based
  findings, the observed count against its frozen predicted range;
- the **suspected loop** — physician, embryologist/ICSI, or culture conditions — labelled
  explicitly as a hypothesis;
- **what else was going on** in that cycle's culture window: the active media and
  consumable lots and whether a lot changed mid-window; the embryologist and physician; the
  incubator environmental load, once available;
- the arrest-pattern flags for the cycle;
- what the relevant consensus document recommends checking, quoted with its source;
- the state of the chart itself: current score, the threshold, and an estimate of when the
  shift began.

**A dashboard**, rebuilt on every run, organised as:

1. **Status** — every chart on one screen, current alerts, traffic-light summary.
2. **Culture-conditions loop** — charts for blastocyst and good-blastocyst rates, envelope
   coverage, and on the same time axis the media lot periods and the incubator episodes
   (with an explicit marker showing where the log coverage currently ends).
3. **Embryologist loop** — fertilisation rate chart, degeneration and abnormal-fertilisation
   rates, and a comparison across operators that accounts for how many cases each handled.
4. **Physician loop** — the same, for retrieval and maturity metrics.
5. **Model audit** — envelope coverage by stage, prediction error, disagreement between the
   two blastocyst forecasts, and the date of the last calibration.
6. **Alert log** — the full history including closed alerts and what the investigation
   concluded.

### 6.4 The false-alarm budget is a management decision, not a statistical one

This is the design decision we consider most consequential, and it inverts the usual order
of work.

The usual approach is to take threshold values from the literature and then observe how
often the system fires. We did the opposite. **The acceptable alarm frequency was decided
first, from the cost of being wrong**, and the thresholds were then computed to deliver it:

| Level | What it costs when wrong | Target frequency, whole system, when nothing is actually wrong |
|---|---|---|
| watch | Minutes of attention | **≤ 2 per month** |
| alert | Hours of work, possible lot quarantine, erosion of trust | **≤ 2 per year** |

The thresholds themselves are then found by simulation: the system builds a model of the
laboratory operating normally — its real volume, its real base rates, and the real
cycle-to-cycle variability measured in its own data — runs many simulated years, and solves
for the threshold values that produce exactly the target alarm frequency. Nothing is taken
from a textbook.

Three consequences follow, and each is enforced in the software rather than left to
discipline:

- **The targets are joint across all charts, not per chart.** Adding a fourth chart
  automatically tightens the thresholds on the other three at the next calibration.
  Otherwise, quietly expanding the dashboard would quietly multiply the false alarms.
- **Fewer charts is usually the right answer.** At our volume, the arithmetic is
  unambiguous: seven charts would require each one to run about three and a half years
  between false alarms, which forces the thresholds so high that real shifts are caught
  late. Three charts require about eighteen months each. Cutting charts lowers *both* the
  false-alarm rate *and* the threshold — one of the rare cases where both sides of a
  trade-off improve together. We run three: one per loop.
- **Until calibration has been performed, the system physically cannot issue an alert** —
  only watch. This gate is released by the calibration process writing its result, never by
  hand. It exists to prevent the single most expensive failure mode: a media lot pulled from
  service on the strength of an uncalibrated threshold.

### 6.5 Corroboration, and why it has to be relative

An alert requires more than one chart crossing a line. The system ranks corroborating
evidence by how independent it is of the biology that produced the signal:

| Source | Rank | Can it escalate watch to alert? |
|---|---|---|
| Unusual incubator environmental load in the cycle's culture window | Strong — physically independent | Yes |
| A media or consumable lot change within the window | Strong — physically independent | Yes |
| Another metric in the same loop | Moderate — correlated through shared biology | No |
| The same metric on a second type of chart | **None — this is not corroboration** | No |

The last row is there because it is a genuinely easy mistake to make. Two different charts
built on the same series of numbers agreeing with each other looks like two independent
witnesses and is not — they are one witness speaking twice.

The more surprising finding concerns the first two rows. Our first implementation treated
the *presence* of a critical incubator episode, or the *presence* of a lot change, as
corroboration. Measured against 49 cycles covered by incubator log exports, that rule was
useless:

- 3,113 incubator episodes across those cycles, of which 1,084 were classified critical —
  roughly 64 episodes and 22 critical episodes **per cycle**;
- **100% of cycles had at least one critical episode**;
- 39% of cycles had a lot change.

An indicator present in every single cycle corroborates nothing. Escalating on it would
have promoted every watch to alert — precisely the opposite of the intent.

The cause was not a parsing error. Over six days across six chambers, opening a lid is
routine embryology work, not an incident, and the gas excursion that follows each opening is
physics, not a fault. A severity classification designed for forensic review of one
selected cycle by a human reader is simply not usable as an automatic binary flag.

The rule was therefore rewritten to match the rest of the system's logic: **corroboration is
comparative, never absolute.** A cycle's environment corroborates an alert only if its load
is unusual *relative to other cycles* — for example, in the top decile of the distribution.
In our data the median cycle accumulates 103 critical minutes and the top decile begins at
1,916. Under the relative rule, 7 of 49 cycles (14%) qualify as an unusual environment,
compared with 49 of 49 (100%) under the absolute rule. The same logic applies to lot
changes: a routine change is not evidence; a change falling exactly inside a period of
declining yield, or an unusual number of changes, is.

### 6.6 The check that prevents the worst kind of mistake

The most damaging error this class of system can make is to declare a laboratory problem
where none exists. There is a specific pattern that signals this risk: **cycles falling
short of their predicted envelope at every stage simultaneously, including stages that occur
before culture even begins.**

Culture conditions cannot affect oocyte maturity. If the shortfall appears there too, the
explanation is elsewhere, and the system investigates three hypotheses in a fixed order
before it will consider the laboratory:

1. **The patients changed.** A cohort with genuinely poor prognosis under-delivers against
   the envelope at every stage, and that is a true statement about the patients. Tested
   first, by comparing the distribution of patient characteristics — age, ovarian reserve,
   antral follicle count, attempt number, sperm source — in the alerting window against the
   reference period, together with an out-of-distribution signal that exists precisely for
   the case of a patient outside the model's training domain.
2. **The model drifted.** Characteristic pattern: patient characteristics have not shifted,
   coverage is below its measured baseline, prediction error is rising, and the last
   calibration is old. The conclusion is a model audit, not a laboratory review.
3. **A laboratory-wide factor.** Considered only once the first two have been excluded.

While that review is unresolved, the coverage check **blocks escalation to alert** — open
findings stay at watch. It does not silence them and does not delete them from the log.

### 6.7 What happens when something fires

| Level | Response |
|---|---|
| **watch** | Appears on the panel. Reviewed at the next routine QC touchpoint. No one is called. |
| **alert** | Triggers a structured investigation: incubator logs for the window, media and consumable lots in use, air quality records, recent protocol or workflow changes, handling review. The alert card supplies the window, the suspected loop, the co-occurring context and the consensus guidance on what to check. |
| **Either** | The outcome of the review is written back into the alert log. |

That last line is not administrative housekeeping. Without recorded outcomes the system
accumulates an unread list, and — more importantly — there is never any data with which to
answer the only question that matters in the long run: *when it fired, was it right?*

### 6.8 The slow confirming loop

Separately from everything above, the system compares the predicted probability of pregnancy
for each cycle against what actually happened.

This loop raises no alerts, ever. Outcome depends on the transfer, the endometrium and a
dozen factors outside the laboratory, and treating it as a laboratory control signal would
be misattribution by construction. Its role is to confirm or contradict the trends seen in
the fast loop over a much longer horizon. When there are too few outcomes to say anything,
it says so rather than producing a number. At present it is explicit that it has not
matured: the reference period contains 6 recorded outcomes against the 20 required.

---

## 7. The role of the AI component — and its boundaries

Because "AI" covers a wide range of things, it is worth stating exactly what the models do
here and, more usefully, what they are not permitted to do.

### 7.1 What the models contribute

**A per-cycle expectation.** The predictive models translate a patient's clinical profile
into an expected yield at each stage of the funnel, expressed as a range. This is the
component that makes case-mix adjustment possible at all — it is what allows the system to
say "this cycle produced few blastocysts, and few blastocysts were expected" rather than
counting it as a shortfall.

**A second, non-drifting opinion.** A mechanistic cascade model provides an expected level
for each transition that is derived from published biology rather than from our own history.
Because it does not move when the laboratory moves, it can see a slow drift that the
laboratory's own reference period would absorb.

**Explanation, not decision.** A language model assembles the human-readable narrative of an
alert card from a package that has already been computed. It writes the explanation; it does
not decide whether anything fired.

### 7.2 What the models are not permitted to do

**They do not supply thresholds.** Every number used in a comparison comes from code — from
the structured consensus definitions or from the calibration output. Retrieval over the
consensus documents supplies wording and quotations for the alert card and takes no part in
the decision of whether something fired. Without that separation, an invented threshold
eventually reaches a card.

**They do not decide alarm levels.** Level assignment is deterministic code operating on the
chart state and the corroboration rules.

**They do not see patient identity.** Pseudonymisation happens at ingest, before any
computation. No model prompt receives names, dates of birth, document identifiers or record
keys.

**Their discrimination metrics are not used in the interpretation.** On cohorts of this size,
the standard model-quality scores are statistically indistinguishable from noise, and
reporting them would lend a false precision to the whole exercise.

### 7.3 An honest negative result about the AI component

We attempted, and abandoned, the most sophisticated use of the model: adjusting each chart
by the model's individual prediction for that cycle, rather than by a single laboratory-wide
base rate. In principle this should absorb patient heterogeneity and allow the thresholds to
be lowered.

Measured on our data, it made things substantially worse — the cycle-to-cycle heterogeneity
the charts had to contend with roughly tripled to quintupled rather than falling.

The cause is not poor prediction. The model predicts well: its central estimate correlates
with the actual count at 0.86 to 0.97 across stages, with a bias of less than one embryo.
The cause is a mismatch between what the model predicts and what a control chart needs:

| Stage | Correlation with the actual **count** | Correlation with the actual **rate** |
|---|---|---|
| Mature oocytes | 0.91 | 0.13 |
| Normally fertilised | 0.97 | −0.08 |
| Blastocysts | 0.86 | 0.16 |

The high correlation on counts is largely an artefact of the denominator: both the actual
value and the prediction rise with the number of oocytes, so "the model predicts well" here
means, to a large degree, "the model knows how many oocytes there were". A control chart
needs the **rate** — what proportion of *this* patient's material will survive to the next
stage — and that information is not present in the forecast.

We also implemented and tested a more careful formulation that uses the *width* of the
predicted range rather than only its midpoint. It removed the numerical instability but did
not recover the missing information, which confirmed the diagnosis: individual risk that is
not in a forecast cannot be extracted from that forecast by any transformation.

Three practical conclusions:

1. The charts run on the laboratory's base rate, which is the configuration for which the
   thresholds were calibrated and verified. The individual-risk adjustment is gated off —
   for a demonstrated reason, not a precautionary one.
2. The residual cycle-to-cycle variability is not an unexplained case-mix effect awaiting a
   better adjustment. It is either genuine biological variation or heterogeneity absent from
   the model's inputs. Lower thresholds should not be expected from this direction.
3. The envelope remains valuable for what it already does — the independent
   actual-versus-predicted-range QA layer — and the machinery for individual adjustment
   remains implemented and tested, so that it can be switched on if a model that predicts
   rates ever becomes available.

This result is documented as prominently as the positive ones. A monitoring system whose
failures are not recorded is not auditable.

---

## 8. What the implementation measured

All figures below come from our own laboratory data and calibration runs. They describe the
system as built, not a target.

### 8.1 Scale

- 91 cycles in the source table, December 2025 – August 2026.
- Roughly **10 retrievals and 70 fertilised oocytes per month** — about 120 cycles and 840
  embryos a year. This is four times smaller than the first design assumption, and it is
  the single number that most shaped the final configuration.
- Declared reference period: 1 May 2026 onward — 37 cycles, 312 fertilised oocytes.

### 8.2 Baseline laboratory performance

Across 77 cycles after protocol exclusions (630 fertilised oocytes), against consensus
competency and benchmark values:

| Metric | Value | 95% CI | Level reached |
|---|---|---|---|
| OCCR — oocytes per follicle | 0.758 | 0.735–0.780 | competency |
| MIIR — maturity | 0.766 | 0.739–0.791 | competency |
| FR — fertilisation | 0.807 | 0.777–0.833 | **benchmark** |
| ClR — cleavage | 1.008 | — | benchmark |
| BlR — blastocyst rate | 0.556 | 0.517–0.594 | competency |
| TGBDR — good blastocyst rate | 0.452 | 0.414–0.491 | **benchmark** |
| DegR — degeneration (upper control) | 0.054 | 0.040–0.072 | competency |
| 3PNR — abnormal fertilisation (upper control) | 0.004 | 0.002–0.010 | benchmark |

Two of these need a note, because both look like errors and neither is one.

*Cleavage rate above 100%* reflects the timing of the fertilisation check — some zygotes
were not observed inside the pronuclear visualisation window and are therefore counted at
cleavage but not at 2PN. It is an observation about timing, not corrupted data. The system
is explicitly forbidden from describing it as a data integrity problem, and it does not
compute a confidence interval for such a row, because the interval is mathematically
undefined there.

*Blastocyst rate at competency while good-blastocyst rate is at benchmark* means the
laboratory loses proportionally more at the transition to blastocyst but obtains a high
proportion of good-quality embryos among those that get there. This asymmetry is exactly why
the loops are kept separate rather than collapsed into a single quality score.

### 8.3 Calibration result

Three charts — one per loop — calibrated on the reference period:

| Chart | Base rate | Observations/year | watch line | alert line | Median time to alert on a real, sharp shift |
|---|---|---|---|---|---|
| FR (embryologist loop) | 0.819 | 1,337 | 1.22 | 7.44 | 1.2 months |
| BlR (culture loop) | 0.590 | 1,095 | 1.78 | 10.83 | 1.7 months |
| MIIR (physician loop) | 0.771 | 1,734 | 1.73 | 10.01 | 1.1 months |

Across the whole system, when nothing is actually wrong: **24.0 false watch findings per
year** against a target of 24, and **1.97 false alerts per year** against a target of 2.

On detection speed, stated honestly:

- A sharp deterioration — the odds of success halved — reaches watch within days of
  accumulated work and alert in roughly one to two months.
- A slow drift of a few percentage points takes three to four months to reach alert. This
  is the resolution limit of the method at 70–140 observations a month, not a tuning defect:
  at that volume a five-point shift is not statistically separable from noise any faster.
  The only two levers are more data or a looser false-alarm target, and both belong to the
  laboratory director, not to the software.

### 8.4 Verification that calibration and live operation measure the same thing

Simulation alone is not sufficient evidence, so the calibrated system was replayed over the
full history. Within the reference period the replay produced **21 watch episodes per year**
against 24 predicted — calibration and runtime agree. Outside the reference period the rate
is higher, at 26 per year, and those are not false alarms: the charts are correctly
reporting that the laboratory then differed from what has now been declared normal.

Getting to that agreement required finding and fixing three discrepancies between the
calibration model and the live implementation, which together had inflated the observed
firing rate roughly threefold:

- **The two sides were counting different events.** The simulation counted threshold
  crossings with a statistic reset; the live system resets only on an alert. Now both use
  one shared implementation, so they cannot diverge again.
- **The order of embryos within a cycle was creating an artefact.** The live system was
  processing each cycle as "all successes, then all failures", which produces a maximal
  excursion at the end of every cycle. Embryo order within a cycle is not observable in
  principle — the outcomes become known simultaneously at closure — so the chart now makes
  a single combined update per cycle. The statistical power is unchanged; the artificial
  within-cycle peaks are gone.
- **Cycle-to-cycle variability was larger than the simple model assumed.** Success rates
  vary between patients by more than random chance alone predicts — measured at roughly two
  to three times the simple expectation, most strongly for blastocyst rate. The charts were
  seeing variability that the threshold model did not contain, and were therefore crossing
  more often than predicted. The calibration now reproduces that measured variability.

### 8.5 Envelope quality assurance

The predicted ranges were calibrated on 5,205 actual-versus-predicted observations pooled
across four laboratories. Two findings changed the specification:

- **12.1% of predicted intervals were degenerate** — narrower than a single embryo. On a
  counting quantity, a band that narrow cannot achieve its nominal coverage, because the
  discreteness of the count exceeds the resolution of the interval. Coverage in those cases
  was 0.414, against 0.812 elsewhere. Such stages are now flagged and excluded from the
  residual layer. Including them would have condemned about one cycle in eight to a
  permanent false "outside the range".
- **The reference coverage is measured, not assumed.** It is 0.812 overall, but ranges from
  0.759 at the fertilisation stage to 0.950 at vitrification. Comparing against the nominal
  0.800 would have produced a permanent false signal at one stage and permanent blindness at
  another.

On our own laboratory's replay, coverage by stage was 0.836 for mature oocytes (+4.2 points
against its measured baseline), 0.720 for fertilisation (−3.9), 0.764 for the early
blastocyst forecast (−5.9), and 0.682 for the post-fertilisation blastocyst forecast (−14.1,
statistically significant). That last one is flagged, and — following the rule in section
6.6 — it blocks escalation to alert until patient mix and model drift have been examined,
rather than being reported as a culture-conditions finding.

### 8.6 Retrospective replay

Replaying the full 91-cycle history through the calibrated system: 89 closed cycles
evaluated, of which **75 were clear, 13 raised watch and 1 reached alert**. Every finding
carried the same verdict — *a trend, not a failure*: a change relative to the laboratory's
own history while remaining inside consensus limits, with the recommendation to examine
patient mix and stimulation rather than the culture protocol.

That verdict distribution is itself the design working. Without the three-frame structure of
section 5.1, those fourteen findings would have been fourteen laboratory investigations.

---

## 9. What the system deliberately does not do

Stating the boundaries is part of the design, not a disclaimer appended to it.

- **It does not evaluate an individual.** A finding in the embryologist loop is a reason to
  review a process, not a person. The wording in every artefact is "a reason for additional
  review". The loop structure exists to localise a cause, and a cause is a step in a
  workflow.
- **It does not claim causation.** An incubator episode coinciding with a drop in yield is a
  candidate explanation. Establishing whether the association is real is a separate piece of
  work on accumulated data.
- **It raises nothing before cycle closure.** Deliberately, as a fixed rule.
- **It does not treat a drop that stays inside consensus limits as a failure.** It reports it
  as a trend and points at patient mix first.
- **It does not compute a confidence interval where one is undefined** — for instance where
  a numerator legitimately exceeds its denominator.
- **It does not report model discrimination scores**, which at this cohort size would be
  indistinguishable from noise.
- **It does not treat an absent environmental record as an absence of environmental events.**
- **It is not a medical device.** It produces observations and reasons to investigate.

There are also two acknowledged limits in the data itself:

- **Incubator chamber is not recorded per cycle.** Our incubator has six chambers, but the
  cycle record does not say which one a cycle used, so an environmental episode is
  attributed to the whole incubator. This is the main limit on the system's resolving power.
  Adding a single column to the retrieval record would enable chamber-level localisation
  after about a quarter of accumulation.
- **Environmental data arrives weekly, not continuously.** A culture-loop investigation is
  therefore often two-part: the biological half immediately, the environmental half after
  the next export. This is designed for, not a malfunction. A continuous environmental
  monitor — minute-by-minute control charts on temperature and gas, detecting a deviation
  *before* the biological outcome — is planned and will attach to the existing escalation
  logic as one more corroborating source, without redesign.

---

## 10. Deployment, portability and data protection

**Two environments, kept unmistakably separate.** Development runs on a draft workbench;
live operation runs on the laboratory machine. Every artefact — every file, every chart,
every alert card — carries a stamp of which environment produced it, and the draft
environment cannot issue an alert at all. A draft report mistaken for a laboratory report is
exactly the class of error the system exists to prevent, so it is prevented structurally.

**Method and local settings are separated.** The method — the chart mathematics, the target
alarm frequencies, the checkpoint rules — is common to every laboratory and lives in one
shared configuration. Everything laboratory-specific — column names in the export, the
declared reference period, which charts are active, local conventions — lives in a
laboratory profile. Bringing up a second laboratory requires a new profile, not a code
change.

**What transfers and what must be recomputed** is defined explicitly, because getting this
wrong is a quiet and serious error:

| Quantity | Source | Why |
|---|---|---|
| Chart mathematics, target alarm rates, checkpoint rules | Shared | It is the method; it does not depend on the laboratory |
| Consensus thresholds | Shared | Universal by definition |
| Envelope scale and coverage | Pooled across available laboratories | It is a property of the *model*, not of any laboratory |
| Base rates | **The laboratory's own data only** | Another laboratory's blastocyst rate is not yours in any approximation |
| Chart thresholds | **The laboratory's own data only** | They depend on its volume and variability |
| Which charts are active | Laboratory profile | It depends on volume; each extra chart tightens the rest |

Calibration is invalidated — and the system reverts to watch-only — whenever the system
moves to another machine or laboratory, the forecast model is retrained or replaced, the
laboratory's volume changes materially, or a chart is added.

**Data protection.** The source table contains full names, dates of birth and identity
document numbers for both partners. Pseudonymisation is applied at ingest, before any
computation; downstream, no personal data exists. No language model or automated agent
receives names, dates of birth, partner details or record identifiers — only a pseudonymous
cycle identifier and an approved list of analytical features. Re-linking a pseudonym to a
patient happens only in local code at the moment a final internal artefact is produced, and
never inside a model's context. Patient data, the pseudonym crosswalk, access tokens, model
weights and calibrated thresholds are all excluded from version control.

---

## 11. Current status and what comes next

**Complete and verified**

- Data pipeline from the retrieval table into the local database, with pseudonymisation at
  the boundary and idempotent daily synchronisation.
- Envelope registration and freezing, with reproducible retrospective registration for the
  historical cohort.
- Baseline measurement: laboratory KPIs, in-control parameters, envelope calibration.
- Threshold calibration by simulation against the declared alarm budget, with the result
  verified by full-history replay.
- The live daily chain: evaluation, charts with persistent state, cause localisation,
  context gathering, verdicts, alert cards and the dashboard.
- Weekly environmental enrichment as a replayable, deterministic pass.
- Portability: laboratory profiles, environment gating, calibration invalidation rules.

**Open**

- **Validation against known incidents.** A register of the laboratory's actual historical
  incidents does not yet exist, and without it there is no way to answer "would the chart
  have caught what happened?". False-alarm frequency is already measured; sensitivity to
  real events is not. Tuning a threshold to fit one known incident is explicitly ruled out —
  that is fitting to a single observation.
- **Prospective observation** — how many findings occur, and how many are substantiated by
  review. This requires the alert log to be closed out consistently by a named person;
  without that, there is no feedback and no validation data.
- **The central hypothesis**, tested on accumulated weekly exports: do environmental
  episodes inside a culture window actually associate with shortfalls at the blastocyst
  stages?
- **A continuous environmental loop**, once incubator data can be streamed rather than
  exported weekly.
- **Chamber-level localisation**, conditional on adding one column to the retrieval record.

---

## 12. Glossary

**Alert** — the upper alarm level. Initiates a structured investigation.

**Base rate** — the laboratory's normal success proportion for a metric during its declared
reference period; the level the charts treat as "as usual".

**Case-mix** — the composition of the patient population: age, ovarian reserve, attempt
number, sperm parameters. A change in case-mix changes laboratory output without any change
in laboratory performance.

**Competency / benchmark** — two levels defined by laboratory consensus documents. Competency
is the minimum acceptable level; benchmark is the level a well-performing laboratory should
achieve.

**Control chart** — a device that monitors a process over time and signals when its behaviour
departs from the expected. Here it is a cumulative score that rises as shortfalls accumulate.

**Corroboration** — independent evidence coinciding with a chart signal. Only physically
independent sources — environmental load, lot changes — can escalate a watch to an alert.

**Envelope** — the range of outcomes predicted for a cycle before culture begins, frozen at
the day of retrieval and never recomputed.

**In-control** — the state of a process operating normally. Alarm frequencies quoted "when
nothing is wrong" refer to this state.

**Loop** — a group of metrics sharing a plausible cause: physician, embryologist/ICSI, or
culture conditions.

**Reference period** — the interval the laboratory director declares to be its normal
operating state. It sets the centre of the relative frame.

**Replay** — recomputing the entire history through the current configuration, to see what
the system would have said.

**Residual** — the gap between what actually happened and what was predicted, expressed
relative to the width of the prediction.

**Watch** — the lower alarm level. Appears on the dashboard; no immediate action.

---

*This document describes a research and quality-monitoring tool developed in-house. It is
not a medical device. It produces observations and reasons for review — not diagnoses, not
clinical instructions, and not assessments of individual staff.*
