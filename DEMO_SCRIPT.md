# NirmanAI — Demo Video Script

**Runs 9:15 at a normal presenting pace** (hard limit 10:00 — see
[Timing](#timing--measured-not-estimated) for what to cut if you speak slowly).
**Format:** 1920×1080 screen recording with voice-over, browser at ~90% zoom.

Every figure below is what the app actually outputs. The order book is seeded
deterministically, so **you will see these exact numbers.**

---

## Before you record

- [ ] `python setup.py` finished with `[OK] All models loaded`
- [ ] `streamlit run app.py` open at `localhost:8501`
- [ ] Masthead shows green **`● Models live`** — if amber, stop and fix
- [ ] Scenario set to **"Monsoon crunch — Patna, Bihar (July)"**
- [ ] **Pre-run the simulator once**, then reload — this warms the cache so it
      returns instantly on camera
- [ ] Notifications off, other tabs closed, bookmarks bar hidden

**Skip tab 2 (Check an Order).** It repeats what tab 1 already showed.

---

## 1 · The problem — 0:00 to 0:55

**Screen:** the dashboard's top band. Don't click yet.

> Two numbers define construction in India. Seventy-seven percent of projects run
> late. Twenty to thirty percent of construction material is wasted on site —
> about one and a half lakh crore rupees a year.
>
> What makes both so hard to fix is that you find out afterwards. A site manager
> discovers the cement is late when his crew is already standing around. He
> discovers the wastage when the store runs dry mid-pour, and now he's paying
> emergency prices.
>
> The information needed to see both coming already exists. Supplier track
> record. The monsoon calendar. State logistics quality. Festival shutdowns. Crew
> skill and site supervision. Nobody puts it together.

---

## 2 · What NirmanAI is — 0:55 to 1:35

**Screen:** same band, point at the four cards.

> NirmanAI is a supply-risk console for Indian construction sites. Three things,
> in the order a site manager needs them.
>
> It scores every open purchase order for delay risk — probability, a calibrated
> day range, and the reasons behind it.
>
> It sizes the wastage buffer per material, so the quantity you order is right the
> first time.
>
> And it simulates the whole project ten thousand times, to show what one late
> delivery does downstream.

*(point at the "Why trust it" card)*

> This card is on the front page on purpose — our accuracy, and next to it, the
> theoretical ceiling. I'll come back to why that second number matters.

---

## 3 · Risk Radar — 1:35 to 3:05

**Screen:** tab **1 · Risk Radar**.

> A residential tower in Patna, Bihar, procuring in July. Peak monsoon. Sixteen
> open purchase orders.

*(point across the KPI row — pause here, let it land)*

> **Eleven of sixteen are more likely than not to arrive late. Four are critical.
> Forty-three lakh rupees of order value at risk, and eighty-six crew-days of
> schedule exposure.**
>
> That last number turns risk into money — delay probability times expected days
> late, summed across the book. Eighty-six crew-days is what this site loses this
> month if nobody acts.

*(scroll to the alert cards)*

> **River Sand from Jharkhand into Bihar. Eighty-eight percent chance it slips,
> about fifteen days if it does.**
>
> This line — "Model's reasoning" — is not a hand-written rule. These are SHAP
> values computed for this specific order: river sand degrading in wet transit,
> ninety percent monsoon intensity, high corridor humidity.
>
> Then the action: split the order, move half to a backup, pull the date forward.

*(scroll to the table)*

> Below is the full book — every row a live model call, with each order's biggest
> risk driver in the last column.

---

## 4 · Proof it's real — 3:05 to 4:05

**The most important minute in the video. Do not cut during the switch.**

> A fair thing to be sceptical about: is this actually running, or is it a
> good-looking mock-up? So let me change the project.

*(Sidebar → **"Dry season — Ahmedabad, Gujarat (February)"**. Let it re-render on camera.)*

> Same product. Commercial complex in Gujarat, February. Dry season, best
> logistics in the country.
>
> **Zero of sixteen at risk. Zero critical. Schedule exposure drops from
> eighty-six crew-days to three.**
>
> Nothing is hardcoded. All sixteen orders were just re-scored against a different
> state, month, suppliers and distances.

*(Switch to **"Festival window — Lucknow, UP (October)"**)*

> One more, because monsoon isn't the only failure mode. Metro depot in Uttar
> Pradesh, October. **Thirteen of sixteen at risk. Eight critical. One and a half
> crore.**

*(point at the top alert's reasoning line)*

> And the reason: **"Order window overlaps a festival shutdown."** Diwali and Chhath
> close plants for a week and trucking capacity collapses. A distinctly Indian
> failure mode the model learned because we encoded the festival calendar.

*(Switch back to **Bihar / July** before continuing.)*

---

## 5 · Wastage and cost — 4:05 to 5:05

**Screen:** tab **3 · Wastage & Cost**.

> Delay is half the problem. The other half is what gets wasted once material is
> on site.
>
> Same bill of quantities, scored against this site's real conditions —
> semi-skilled crew, poor supervision.
>
> **Eight lakh forty-three thousand rupees of material this project throws away.**

*(point at the insight callout)*

> And here's my favourite thing it does. **River Sand has the worst wastage rate —
> thirty-one percent. But TMT Steel loses more money: three lakh forty-four
> thousand, at only nine percent — because steel is expensive.**
>
> Manage by percentages, you chase the sand. Manage by rupees, you chase the
> steel. The system tells you which list to work from.

*(scroll to the supervision ladder)*

> And this is the business case. Each row is a fresh model run, changing only
> supervision quality. **Poor to excellent saves three lakh ten thousand rupees on
> this one bill of quantities.** That's how you justify hiring another site
> engineer — priced by the model.

---

## 6 · The simulator — 5:05 to 6:35

**Screen:** tab **4 · Project Simulator**.

> But the real question isn't about one order. It's: does my project finish on
> time? One late cement delivery delays the pour, which delays the masonry, which
> delays everything after it.

*(click **Run simulation**)*

> So we simulate the entire activity network ten thousand times.

*(let it land — pause)*

> **Seventeen percent chance this project finishes on time. Most likely two
> hundred sixty-one days against a two-thirty plan. Worst ten percent, two
> hundred seventy-two.** That took about a second.
>
> The naive version calls the model inside the loop — three hundred thousand
> calls, minutes of compute. We score it once per material per month instead —
> about a hundred and thirty calls — to build a calibrated risk profile, then
> sample from those. **The machine learning still sets every probability.**

*(scroll through distribution, activities, materials)*

> The full distribution of where this project lands. Then the diagnosis: which
> activities slip, which materials cause it, and the single points of failure —
> activities depending on one material with no fallback.

---

## 7 · Plan and honesty — 6:35 to 7:40

**Screen:** tab **5 · Plan & Report** → **Generate procurement plan**.

> Turning that into something a buyer acts on this morning. Each material gets a
> **risk-weighted** buffer — probability times expected days late. A
> ninety-percent-risk cement order gets a real buffer; a five-percent tile order
> gets almost none.

*(scroll to supplier allocation and the Pareto chart)*

> On supplier allocation — this is where prototypes overclaim. Assume two
> suppliers fail independently and the maths says splitting removes ninety-seven
> percent of your risk. That's wrong: the same monsoon hits both. **We model that
> correlation, so splitting removes about half the stock-out risk, and no more.**

*(open the honesty expander)*

> And I want to be direct. **The purchase orders here are synthetic** — we have no
> ERP to read from. They're generated from the selected project, then scored by the
> real models. The training data is synthetic too, built from published benchmarks:
> CIDC wastage ranges, IMD monsoon profiles, state logistics indices, the festival
> calendar.
>
> The models are real. The seventy-one suppliers are a curated database. And this
> panel is in the product, not just in the video.

---

## 8 · Why the accuracy number is the point — 7:40 to 8:35

**Screen:** the "Why trust it" card or the methodology expander.

> Which brings me back to that number. Our delay classifier scores **AUC 0.797**.
> An earlier version reported 0.918 — we deleted it, because we found where it
> came from.
>
> The data had a traffic feature recorded **during** transit: heavy traffic meant a
> hundred percent delayed. But congestion seen while the truck is already late is a
> *consequence* of the delay, not something a buyer knows at order time. The model
> was reading the answer.
>
> We rebuilt the generator to forecast traffic at order time. Accuracy dropped.
> That's the honest number.
>
> And because a delivery either slips or it doesn't, there's a hard ceiling on what
> any classifier can reach. We compute it: **0.829**. We score 0.797 — **ninety
> percent of the learnable signal.**
>
> Every prediction also ships with a conformal interval — ninety percent target
> coverage, ninety point three measured. Because telling a site manager "your
> cement arrives on the fifteenth" with no range is how crews end up idle.

---

## 9 · Close — 8:35 to 8:55

> NirmanAI turns information Indian construction sites already have into decisions
> they can act on this week: which orders to chase, how much to actually order, and
> where the schedule really breaks.
>
> Honest about what it knows, how confident it is, and what's still synthetic.
> Thank you.

---

## The exact numbers you'll see

### Monsoon crunch — Patna, Bihar (July)
| | |
|---|---|
| Open orders | 16 |
| Likely to slip | **11** (69%) |
| Critical | **4** |
| Value at risk | **₹43.11 L** |
| Schedule exposure | **86 days** |
| Top alert | River Sand, Jharkhand → Bihar, **88%**, ~15 d if late |
| Simulator | **17%** on time · plan 230 d · likely 261 d · P90 272 d · Critical |
| Wastage overrun | **₹8.43 L** · weighted **18.7%** |
| Worst by cost / by % | TMT Steel ₹3.44 L @ 9.2% / River Sand 30.9% |
| Supervision saving | **₹3.10 L** (Poor → Excellent) |
| Stock-out risk cut | **50%** (ρ = 47%) |

### Dry season — Ahmedabad, Gujarat (February)
| | |
|---|---|
| Likely to slip | **0 of 16** |
| Critical | **0** |
| Schedule exposure | **3 days** |
| Simulator | **81%** on time · Low |

### Festival window — Lucknow, UP (October)
| | |
|---|---|
| Likely to slip | **13 of 16** (81%) |
| Critical | **8** |
| Value at risk | **₹1.52 Cr** |
| Top alert driver | *"Order window overlaps a festival shutdown"* |
| Simulator | **19%** on time · Critical |

---

## If a number looks different

1. **Check the project name hasn't been edited** — it's part of the random seed.
   Must read exactly `Ganga Riverside — Tower B` (em-dash).
2. **Check the masthead chip is green.** Amber means models didn't load and
   everything shown is a rule-based estimate.
3. If you re-ran `python setup.py`, the data regenerates and simulator figures
   shift a point or two. Say "about seventeen percent", not the decimal.

---

## Timing — measured, not estimated

The script is **1,179 spoken words**. Speech time plus roughly 50 seconds of
silent clicking and page renders:

| Your pace | Speech | **Total video** |
|---|---|---|
| 125 wpm (slow, deliberate) | 9:25 | **10:15** ⚠️ over |
| 140 wpm (normal presenting) | 8:25 | **9:15** ✅ |
| 155 wpm (brisk) | 7:36 | **8:26** ✅ |

Per segment at 140 wpm:

| Segment | Speech | Ends around |
|---|---|---|
| 1 · Problem | 0:46 | 0:50 |
| 2 · What it is | 0:43 | 1:35 |
| 3 · Risk Radar | 1:10 | 2:55 |
| 4 · Proof it's real | 0:58 | 4:00 |
| 5 · Wastage | 1:03 | 5:10 |
| 6 · Simulator | 1:06 | 6:30 |
| 7 · Plan + honesty | 1:11 | 7:45 |
| 8 · Why AUC matters | 1:09 | 9:00 |
| 9 · Close | 0:20 | 9:15 |

### Read this before recording

**Time yourself on segment 1 alone.** It should take about **46 seconds**. If it
takes you over a minute, you are speaking at ~125 wpm and the full video will run
past 10:00 — take the two cuts below before you start.

### Cuts, in order of what to drop first

1. **Segment 6**, the "three hundred thousand calls" paragraph — saves 0:20
2. **Segment 7**, the Pareto/allocation paragraph, keeping the honesty one — 0:20
3. **Segment 4**, the Lucknow festival scenario — saves 0:25

Taking all three brings you to about **8:10** at any pace.

**Never cut** the Bihar → Gujarat switch in segment 4, or segment 8. Those are
the two moments that separate this from a mock-up.

---

## Recording tips

- **Don't cut during the scenario switch.** The judge must see the page re-render
  live — a cut reads as an edit.
- ~140 words per minute. This script is written at that pace; if you naturally
  speak faster you'll land closer to 8:00.
- Fluff a line? Pause two seconds, say it again, cut later.
- Easiest method: record silent screen capture first, then voice-over on top.
- Keep the generated PDF report open in a second tab as a backup visual.
