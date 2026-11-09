# Same Tasks, Same Ceiling - code for our Gemma 4 Developer Agent paper

Code and agent configurations behind our Kaggle Gemma 4 Developer Agent Paper Track writeup
([docs/paper_writeup.md](docs/paper_writeup.md)). Competition data (tasks, snapshots, reference patches) is
not included; download it from the competition page.

## Agents (`agents/`)
| Folder | What it is |
|---|---|
| `submission_v4_fast` | Single Gemma 4 31B agent, heredoc edits, 384-token thinking cap, 5.5 min. Our LoRA adapter (v4) is not included; remove the `adapter:` line to run it on the base model. |
| `submission_v6c6` | Fixer → fresh-context reviewer (`SequentialAgent`), base 31B model, 6.0 min. |
| `submission_v6e_e4b` | Same idea on Gemma 4 E4B with three reviewers; only the last one can submit. |

## Tools (`tools/`)
| Script | Purpose |
|---|---|
| `make_test_nb.py` | Builds Kaggle notebooks that run an agent on fixed task sets and record every LLM call (eval / paired-50 / collect / synthetic modes). |
| `build_sft_data.py` | Turns recorded successful runs into chat-template-rendered SFT sequences with loss spans on model output. |
| `trim_sft_data.py` | Fits SFT data into a token budget. |
| `make_train_nb.py` | Builds the QLoRA training notebook (plain-PyTorch LoRA saved in PEFT format; works without internet). |
| `bug_factory.py`, `factory_worker.py`, `build_synth_tasks.py`, `validate_synth_tasks.py`, `nb_cells/synth_cell.py` | SWE-smith-style synthetic bug factory: AST mutations, hidden failing tests, validation, issue writing with Gemma. |
| `make_hints.py`, `build_gold_demos.py` | The hint (STaR-style) and reference-fix demonstration experiments reported as negative results. |

## License
Apache License 2.0 (see `LICENSE`).
