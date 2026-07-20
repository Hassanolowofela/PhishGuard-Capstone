# PhishGuard Phase 5: Usability Test Plan and Instrument

**Project:** PhishGuard, an interpretable web based tool for phishing and malware email detection
**Course:** MSIT 5910 Capstone Project
**Author:** Hassan Olowofela
**Purpose of this document:** a ready to run, moderated usability test for non technical users, plus the survey and the scoring template used to analyse the results.

Usability for non experts is a stated design goal of PhishGuard, so it must be tested with real users rather than assumed. This plan is deliberately small and practical: it can be run with two to five participants in about twenty minutes each, which is enough to surface the most serious usability problems.

## 1. Objectives

The test answers four questions. Can a non technical person paste an email and get a verdict without help? Do they correctly understand what Safe, Scam, and Malware mean here? Do the plain language reasons and the recommended action actually change what they would do next? Do they trust the result, and is that trust appropriate?

## 2. Participants

Recruit two to five adults who use email daily but have no security or software background. Record only a rough profile for each: age band, how often they receive suspicious email, and self rated comfort with technology on a one to five scale. No personal data beyond this is collected, which matches the privacy stance of the project.

## 3. Setup

The moderator runs the app locally and shares the screen, or the participant drives on their own machine. Prepare three prepared emails in advance: one ordinary legitimate message, one obvious scam, and one borderline message. Do not tell the participant which is which. Ask the participant to think aloud, and record only observations and quotes, not the screen, unless the participant agrees.

## 4. Task scenarios

Give each task in plain language and let the participant work without hints for the first two minutes.

Task 1, first run without guidance. "You received this email and you are not sure about it. Use this tool to check it." Observe whether they find the input fields, paste the content, and start the analysis unaided.

Task 2, interpret a scam result. Using the scam email, ask: "What is the tool telling you, and what would you do next?" Observe whether they read the verdict, the confidence, and the reasons, and whether they would follow the recommended action.

Task 3, interpret a borderline result. Using the borderline email, ask the same questions. Observe whether the confidence score and reasons help them stay appropriately cautious rather than over trusting a single label.

Task 4, explanation value. Ask them to point to the part of the screen that best explains the decision, and whether it made sense.

## 5. Observation metrics (moderator records per task)

For each task the moderator records four things. Task success on a three point scale: completed unaided, completed with one hint, or not completed. Time to first verdict in seconds for Task 1. Error and confusion events, meaning any point where the participant hesitated, clicked the wrong place, or misread the result. Verbatim quotes that reveal a mental model, especially any misunderstanding of Safe, Scam, or Malware.

## 6. Post test survey (System Usability Scale, adapted)

After the tasks, the participant rates ten statements from 1 (strongly disagree) to 5 (strongly agree). The wording is adapted to name the tool but the standard scoring is preserved.

1. I think that I would like to use PhishGuard frequently.
2. I found PhishGuard unnecessarily complex.
3. I thought PhishGuard was easy to use.
4. I think I would need help from a technical person to use PhishGuard.
5. I found the different parts of PhishGuard were well integrated.
6. I thought there was too much inconsistency in PhishGuard.
7. I imagine most people would learn to use PhishGuard very quickly.
8. I found PhishGuard very awkward to use.
9. I felt confident using PhishGuard.
10. I needed to learn a lot of things before I could get going with PhishGuard.

Three short free text questions follow the scale. What was the single most confusing thing? What, if anything, made you trust or distrust the result? What one change would help you most?

## 7. Scoring the survey

The System Usability Scale produces one number from 0 to 100. For the odd numbered items subtract 1 from the rating. For the even numbered items subtract the rating from 5. Add the ten adjusted values and multiply the total by 2.5. Interpret the result with the common benchmark: about 68 is average, above 80 is good, and below 50 signals serious usability problems. Report the mean and the range across participants, not just the average, because with a small sample the spread matters.

## 8. Analysis template (fill after sessions)

| Participant | Comfort (1 to 5) | Task 1 success | Time to first verdict (s) | Understood Safe/Scam/Malware? | SUS score |
|-------------|------------------|----------------|---------------------------|-------------------------------|-----------|
| P1 |  |  |  |  |  |
| P2 |  |  |  |  |  |
| P3 |  |  |  |  |  |
| P4 |  |  |  |  |  |
| P5 |  |  |  |  |  |

**Top usability issues (ranked by how many participants hit them):**

1.
2.
3.

**Quick wins (low effort, high value changes):**

1.
2.

**Overall SUS:** mean __ , range __ to __ , across __ participants.

## 9. Ethical notes

Participation is voluntary and can stop at any time. No email belonging to the participant is required; the prepared samples are used instead. No personally identifying information is stored. These constraints keep the test consistent with the privacy by design principle of the project itself.
