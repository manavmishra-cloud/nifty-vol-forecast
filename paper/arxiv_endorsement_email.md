# arXiv endorsement email — strategy and template

## Why you need an endorsement

arXiv requires first-time submitters in some categories (including
`q-fin.ST`) to be **endorsed** by an existing arXiv author in that
category. This is a one-time gate; once endorsed, you can submit
freely in that category forever.

## How to find a potential endorser

Look for researchers who:
1. Have **at least 2 papers** posted to `q-fin.ST` (or related
   `q-fin.*` subcategories like `q-fin.TR`, `q-fin.RM`)
2. Posted at least one paper to arXiv in the **past 5 years**
3. Have an **available email** (most do via their institutional pages)
4. Work on topics related to your paper — volatility, realized
   variance, machine learning in finance, or Indian markets specifically

Good candidates to investigate (do not email yet — verify they
actually post to arXiv first):

- **Andrew Patton** (Duke) — QLIKE loss developer
- **Peter R. Hansen** (UNC) — Realized GARCH
- **Kim Christensen** (Aarhus) — ML in vol forecasting
- **Antonio Briola** (UCL) — recent arXiv author on LOB DL
- **Sebastian Jaimungal** (Toronto) — algorithmic trading
- **Marcos López de Prado** (Cornell) — ML in finance
- **Rama Cont** (Oxford) — microstructure
- Any author of a paper cited in your `references.bib` who is
  active on arXiv

**How to verify:** go to https://arxiv.org/a/[lastname]_[first-initial]_1.html
(e.g., https://arxiv.org/a/patton_a_1.html) and check if they have
recent q-fin submissions.

## The email template

Below is a template. Personalize the **bracketed parts** before
sending. Send to one or two endorsers at a time — do not mass-email.

---

**Subject:** arXiv endorsement request — q-fin.ST submission on volatility forecasting

Dear Prof. [Last Name],

I am [your full title], an undergraduate researcher at the Vellore
Institute of Technology in India. I am writing to request an arXiv
endorsement for the q-fin.ST category for a paper titled
*"Sequence Models, Not Tabular ML, Beat HAR-RV: A Walk-Forward
Benchmark of Volatility Forecasts on NIFTY-50"*.

The paper benchmarks GARCH(1,1), HAR-RV, XGBoost, LSTM, and a
Transformer encoder on 18+ years of NIFTY-50 daily realized
volatility, evaluated under QLIKE loss and Diebold-Mariano tests.
The headline finding is that sequence models significantly beat
HAR-RV ($DM=+5.78$, $p<10^{-4}$ for LSTM; $p=0.026$ for the
Transformer), while tabular XGBoost on engineered HAR-style
features is statistically indistinguishable from HAR ($p=0.48$).
This refines the often-cited result that "ML cannot beat HAR" --
the limitation is architectural, not classical-versus-modern.

I selected your name as a possible endorser because of your
related work on **[1-2 sentences referencing one of their papers
specifically -- read it first, cite a concrete contribution]**.

The full paper draft is attached, and the source code is publicly
available at https://github.com/manavmishra-cloud/nifty-vol-forecast.
I am happy to share the Overleaf source or a PDF directly.

To complete the endorsement, you would need to log into arXiv
and visit the endorsement page (https://arxiv.org/auth/need-endorsement).
arXiv will email you a confirmation request once I initiate
submission with your name listed.

I appreciate your time and would be grateful for your
consideration.

Best regards,
Manav Mishra
B.Tech in Computer Science and Engineering (Core)
Vellore Institute of Technology, Vellore, India
[your email address]
[github.com/manavmishra-cloud]

---

## Practical tips

1. **Personalize the "I selected your name because" line.** Generic
   emails get ignored. Read one of their recent papers and reference
   a specific finding or method. This single sentence is the
   difference between a 5% and 50% response rate.

2. **Attach the PDF.** Most endorsers want to skim the paper before
   agreeing. Don't make them dig through GitHub.

3. **Send to 2-3 people in batches**, not 10 at once. Wait 1-2 weeks
   between batches. Mass emails look like spam.

4. **Have your arXiv account already created.** Before requesting
   endorsement, sign up at https://arxiv.org/user/. You'll need this
   to receive the endorsement once granted.

5. **What to do if rejected/ignored:** Most endorsement requests are
   ignored, not actively rejected. This is normal. Try a different
   researcher. Don't take it personally.

6. **Backup option: institutional endorsement.** If your university
   has affiliated researchers who already post to arXiv (check VIT
   faculty pages), they can endorse you. This is sometimes easier
   than cold-emailing famous researchers.

7. **Wait until paper is publication-quality.** Don't request
   endorsement on a draft you wouldn't submit today. The endorser
   will look at the paper and judge whether it's serious work.

## Timing

Based on your timeline:
- **Now:** Sensitivity experiments running; paper draft v1 in place
- **Next 2-3 weeks:** Polish paper (Robustness section, additional
  figures, careful proofread)
- **Then:** Identify 2-3 potential endorsers, read one paper from
  each, draft personalized request
- **Submit endorsement requests:** Send to 2 endorsers first
- **Once endorsed:** Submit to arXiv. Typically goes live in 1-2
  business days after passing automated checks.

Target arXiv submission date: **late July or August 2026**.
