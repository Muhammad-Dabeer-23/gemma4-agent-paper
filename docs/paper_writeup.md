# Same Tasks, Same Ceiling: What We Learned Trying to Teach Gemma 4 to Fix Bugs in Five Minutes

### Self-training that didn't move the score, a leaderboard that couldn't tell us apart, and the one change that helped (locally)

## Abstract

Over six weeks we tried to improve an offline bug-fixing agent built on Gemma 4 31B, running on four L4 GPUs with roughly five minutes per issue. Most of what we tried did not work, and we think the reasons are worth sharing. Identical submissions scored anywhere from 0.10 to 0.17 on the public leaderboard, so we stopped trusting single scores and ran every version on the same 50 training tasks instead. Fine-tuning LoRA adapters on the agent's own successful runs made it tidier but never changed which tasks it solved: the base model and three adapters all landed at 10–12 out of 50. Hints, reference-fix demonstrations and stricter prompts didn't help either, and one of them made the agent noticeably worse. Looking at the failures explained why. In 71% of failed runs the agent was already editing the right file; it just guessed the wrong behaviour from a vague issue. The only change that consistently helped was structural: a second agent, starting with a fresh context, that reviews and repairs the first agent's fix before anything is submitted. That took us from 10–12 to 13–14 out of 50. It did not carry over to the hidden test set, and we report that too.

## 1. Why we're writing this

The competition sets up an unusual problem. Instead of a large hosted model with plenty of time, you get one open 31B model, four L4 GPUs, no internet, and twelve hours for about 120 hidden issues. In practice that means around five minutes per issue at about 31 generated tokens per second. One long reasoning step can cost as much time as eight quick shell commands.

We went in with the standard plan for open models: let the agent attempt tasks, keep the runs that pass the tests, fine-tune on them, repeat. We built a small bug factory to get more practice tasks than the 129 provided ones. The training worked in the narrow sense: the loss went down and the adapters loaded fine. The agent just kept solving the same tasks. When we found a forum thread from another team saying almost exactly that ("loss falls a lot, solve rate doesn't move"), we decided the more useful contribution was to find out why.

What this paper offers:

- A simple evaluation habit that made our decisions far more reliable than the leaderboard did.
- A set of negative results on self-training at this budget, with the ablations behind them.
- A breakdown of how the agent actually fails.
- A reviewer pipeline that improved local results, plus the harness quirks that quietly broke our first attempts at it.
- An honest check of whether any of this transferred to the hidden tasks. Mostly it didn't.

## 2. Setup

**The harness.** The official harness (swegemma 0.2.11 and adk-submission 0.2.13) turns a YAML agent definition into a Google ADK agent, serves the model with vLLM across all four GPUs, and runs each task in a sandbox. After the agent finishes, hidden tests are applied to whatever diff it left behind. Edits count even if the agent never calls `submit_patch`. Time, tool-call and turn limits are set per task in `eval_config.yaml`, and if the whole run goes over twelve hours, the submission scores zero. vLLM start-up takes about 15 minutes and each sandbox about 15 seconds, so the real per-task budget is tighter than it looks.

**The model.** Unless stated otherwise we used `gemma-4-31b-it-qat-w4a16-ct`. A typical task took about 27 model calls, with 7.5–9 seconds between them.

**The baseline agent.** One agent, five tools: `run_command`, `read_file`, `submit_patch`, and two code-graph lookups. We dropped the harness's `edit_file` and `write_file` early, after seeing about a third of their calls arrive malformed. The agent edits with a short Python heredoc that checks the text it replaces appears exactly once, which turned out to be much more reliable. The prompt asks for a first edit within eight tool calls.

**How we evaluated.** Every version ran on the same 50 training tasks: 20 we kept out of all training data from the start, plus 30 more from the provided set (fastapi, rich, requests and httpx). Rather than just counting, we compared which tasks each version gained or lost against a reference run. Once we adopted this protocol, we excluded all 50 tasks from every later training set.

## 3. The leaderboard couldn't tell our versions apart

The public leaderboard uses about half of the hidden tasks, so a single task is worth roughly 0.017. We noticed something was off when we resubmitted the same file and got a different score. So we did it on purpose:

| Same files, submitted repeatedly | Public scores |
|---|---|
| v2 adapter, 5 min | 0.17, 0.10, 0.15 |
| v4 adapter, 5.5 min, 384-token thinking cap | 0.15, 0.17, 0.15, 0.15, 0.17 |

A 0.07 swing is about four tasks. With one submission a day, picking the best-looking score mostly means picking the luckiest run. Our local 50-task runs were the opposite: the same configurations solved nearly the same tasks every time, so when a change gained or lost two or three tasks consistently, we could actually believe it.

## 4. Training on our own successes didn't change the outcome

**Getting the data.** We recorded every request and response the agent sent to vLLM during runs on Kaggle. For runs that passed, we re-rendered the conversation with Gemma's own chat template and trained only on the parts the model had generated: its reasoning and its tool calls. Real-task data came from the provided training tasks. Synthetic tasks came from our bug factory, which mutated three repository snapshots (removing a branch, dropping a keyword argument, stubbing out a function body) and kept a bug only if one to five previously passing tests started failing. Those tests were hidden from the agent and used for grading, and Gemma wrote the issue text itself. About a quarter of those were deliberately just a title, because many real issues are. We also dropped runs that took more than ten tool calls to make a first edit.

**Training.** Rank-16 LoRA (alpha 32) on the attention query and output projections and the MLP, trained with 4-bit QLoRA on the unquantized QAT checkpoint. Kaggle's L4 sessions have no internet access, so we couldn't install PEFT and wrote the LoRA layers in plain PyTorch, saving them in PEFT's format so vLLM could load them. One epoch, learning rate 1e-4 with cosine decay, sequences up to 14,336 tokens, about 56–75 tokens per second.

| Version | Training data | Solved (of 50) |
|---|---|---|
| Base model, no adapter | – | 12 |
| v2 adapter | 101 tasks, 1.25M tokens, ~70% synthetic | 10 |
| v4 adapter, 384-token thinking cap | 105 sequences, 0.98M tokens, 68% synthetic | 11 |
| Base model, no adapter, 6.5 min (control) | – | 12 |

The adapters did change how the agent worked. With v4, fewer tasks ran out of time (12 instead of 15), fewer ended with no edit (4 instead of 6), and the first edit came about one call sooner. But the set of solved tasks barely moved, and none of the adapters beat the base model. Another adapter that leaned harder on real tasks (v3) scored 0.13 publicly the same week v2 got 0.17, which, given the leaderboard noise above, tells us nothing either way. Our best explanation is that training on your own successes mostly reinforces what you can already do. Under a five-minute budget there wasn't much left to gain from behaviour alone.

**Thinking budget.** Long responses (over 400 tokens) used 65% of the agent's time, and 79% of those tokens were reasoning. Capping reasoning at 384 tokens instead of 1,024 nudged the result from 10 to 11 and cut time-outs from 19 to 15. Turning reasoning off completely backfired: steps got twice as fast, but the result fell to 7. Fifteen tasks burned through every tool call and twenty ended with no edit at all. A little thinking turned out to be much better than none.

## 5. The agent usually knew where to look

We sorted 118 failed runs by comparing the files the agent touched with the files the reference fix touched:

| What went wrong | Share |
|---|---|
| Edited all the right files, but the fix was wrong or incomplete | 52% |
| Edited some of the right files; the fix needed more | 19% |
| Edited only the wrong files | 15% |
| Never edited anything | 12% |

The tasks we solved had small reference fixes (median 4 changed lines, about one file). The ones we didn't solve needed more (median 15 lines across 2.4 files). Many of the unsolved issues are one line long, often just a pull-request title. A typical example: the issue says an exception in a text-handling method was fixed, the agent finds the method straight away, and then protects against a different exception than the one the hidden test checks for. Finding the code was rarely the problem. Working out what the maintainers wanted was.

## 6. Three ways we tried to teach it more

**Hints.** For 33 tasks the agent had never solved, we added a note to the issue saying which files and functions the real fix touched and which lines were wrong, without giving the new code. That's the idea behind STaR-style rationalisation. Only 2 of the 33 were solved. In 28 of the 31 failures the agent edited exactly the files we pointed it to and still got the fix wrong.

**Showing it the right answer.** For 34 tasks the agent had failed, we kept its own exploration as context (not trained on) and replaced its edit with the reference fix, written in the agent's usual heredoc form and followed by a submit. We checked that every one of these edits reproduced the reference patch exactly. The adapter trained on this (v5) solved 10 of 50, and its behaviour fell apart. It started repeating the fixed sentence we had used in those examples ("I have enough context. Applying the fix."), edited after about six steps, and submitted even when its edit had failed, leaving 12 tasks with no change at all. Thirty-three examples taught it a habit, not a skill.

**Stricter prompts.** We added "list every place the fix needs" and "prove the fix by running the issue's example before submitting". On a 40-task subset the result didn't change (11), and the agent checked its fix after editing in 43% of runs compared with 48% before. It simply didn't follow the new instructions.

## 7. Making verification impossible to skip

If the agent wouldn't verify when asked, we decided to make verification a separate step it couldn't skip. The harness supports a `SequentialAgent` that runs sub-agents in a fixed order:

- **Fixer:** the base model with the usual prompt and tools, except `submit_patch`. It ends by describing its change in one sentence.
- **Reviewer:** the base model again, with `include_contents: none` so it never sees the fixer's reasoning. Its prompt contains only the issue (via the `problem_description` session variable) and a short routine: run `git diff`, run the issue's example and the module's tests in one command, then either submit as is or repair and submit.

Getting this to actually run took two tries, and both failure modes were silent. First, the harness ends a task the moment any agent calls `submit_patch`. In our first version the fixer still had that tool, so the reviewer ran on just 1 of 50 tasks. Second, when we removed the tool but kept the LoRA adapter on the fixer, it couldn't stop. Its reasoning said "I'll stop now", and then it ran the same test command again and again until the tool budget was gone. The adapter had learned that finishing means calling `submit_patch`. Switching the fixer to the base model fixed both problems.

| Version (same 50 tasks) | Solved | Mean time per task | Empty patches |
|---|---|---|---|
| v4 single agent, 5.5 min | 11 | 229 s | 4 |
| Base single agent, 6.5 min (control) | 12 | 220 s | 3 |
| Fixer → reviewer, 6.5 min | 14 | 301 s | 1 |
| Fixer → reviewer, 6.0 min (second run) | 13 | 280 s | 4 |

The control was important. Given the same 6.5 minutes, a single agent finished in 220 seconds on average and never used the extra time. In its first run, the reviewer pipeline solved everything the control solved plus two more tasks. To be fair to the single agents, though: across all nine of our single-agent runs, 15 different tasks were solved at least once, and only one task was ever solved by the pipeline alone. So the reviewer doesn't unlock new abilities. It makes the borderline tasks pass more often by giving each fix an independent second look.

**Did it transfer?** At 6.0 minutes the pipeline projects to about 10.1 hours for 120 tasks, and it finished within the limit on Kaggle. Its public scores were 0.15, 0.13 and 0.12 (mean 0.133), while the single v4 agent averaged 0.158 over five submissions (0.15, 0.17, 0.15, 0.15, 0.17). Given the noise we can't say it's worse, but the local gain clearly didn't show up. Our guess is that the hidden tasks come from private repositories with different tests and conventions, while our local tasks come from popular public projects the model may partly remember.

**Small models.** After the organisers allowed the smaller Gemma 4 variants, we ran the same pipeline with E4B and E2B. E4B solved 4 of 50 and E2B 2 of 50. E4B was fast enough to finish in about 80 seconds per task, so we added two more reviewers. That brought it to 5 of 50, with fewer empty patches, but no clear jump. On the public leaderboard, the one-reviewer E4B version scored 0.06. The pattern matches the 31B results: extra review passes make the agent more dependable, but they don't raise its ceiling.

## 8. Related work

SWE-bench (Jimenez et al., 2024) made "resolve a real GitHub issue, judged by hidden tests" the standard benchmark for this kind of agent. SWE-agent (Yang et al., 2024) showed how much the agent's tool interface matters, which matches our experience replacing the built-in edit tools. Agentless (Xia et al., 2024) splits the work into fixed localisation, repair and validation phases, which is close to what our reviewer pipeline does. Our training loop follows rejection-sampling fine-tuning and expert iteration (Yuan et al., 2023; Gulcehre et al., 2023), and our hint experiment follows STaR (Zelikman et al., 2022). SWE-Gym (Pan et al., 2024) and SWE-smith (Yang et al., 2025) scale up environments and synthetic bugs for training agents, and our bug factory is a small version of SWE-smith's approach. Self-Refine (Madaan et al., 2023) and Reflexion (Shinn et al., 2023) have a model critique and revise its own work. Our reviewer differs in starting from a clean context and in being enforced by the agent structure rather than requested in the prompt. LoRA (Hu et al., 2022) and QLoRA (Dettmers et al., 2023) made the training possible on our hardware.

## 9. Limitations

Our local benchmark is 50 tasks from four public repositories, and one or two tasks of difference is within normal run-to-run variation; where possible we repeated runs. We have only a handful of public scores per version. The synthetic bugs come from three repositories. We only tried rank-16 LoRA for one epoch, so larger or longer training might behave differently. And most importantly, the improvements we measured locally did not carry over to the hidden private repositories.

## 10. What we'd tell the next team

Measure on a fixed set of tasks and compare which tasks change, not just the total; the leaderboard alone will mislead you. Don't expect training on your own successes to raise the ceiling under a tight time budget, and look at your failures early. Ours said "right file, wrong guess" long before we stopped adding training data. If you want the agent to verify its work, build verification into the structure, because asking in the prompt doesn't work. And plan for the hidden repositories being different from the ones you practise on. The next real gain is probably in helping the model infer what the maintainers intended, for example from the repository's history and existing tests.

## Reproducibility

All of our tooling will be released under the Apache-2.0 licence: trajectory recording, SFT data builder, plain-PyTorch QLoRA trainer, bug factory, notebook generator and analysis scripts. Competition data is not redistributed.

## References

- Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
- Gulcehre, C., et al. (2023). Reinforced Self-Training (ReST) for Language Modeling. arXiv:2308.08998.
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Jimenez, C. E., et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR.
- Madaan, A., et al. (2023). Self-Refine: Iterative Refinement with Self-Feedback. NeurIPS.
- Pan, J., et al. (2024). Training Software Engineering Agents and Verifiers with SWE-Gym. arXiv:2412.21139.
- Shinn, N., et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS.
- Xia, C. S., Deng, Y., Dunn, S., Zhang, L. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. arXiv:2407.01489.
- Yang, J., et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS.
- Yang, J., et al. (2025). SWE-smith: Scaling Data for Software Engineering Agents. arXiv:2504.21798.
- Yuan, Z., et al. (2023). Scaling Relationship on Learning Mathematical Reasoning with Large Language Models. arXiv:2308.01825.
- Zelikman, E., Wu, Y., Mu, J., Goodman, N. (2022). STaR: Bootstrapping Reasoning With Reasoning. NeurIPS.
