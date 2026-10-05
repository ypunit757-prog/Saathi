# Deploying Saathi on Render

**Why Render:** Saathi is one Flask process plus one SQLite file. Render runs that as a normal web service, gives free HTTPS, deploys on every `git push`, and has a persistent disk for the SQLite file. (It is also a hackathon sponsor category: "Best Use of Render".)

**The one thing that changes in the cloud:** Saathi was built to run Gemma through Ollama on a laptop. Render has no GPU and no Ollama, so the deployed app calls a *hosted* Gemma through an API instead. Your notes are then sent to that provider. The code picks the backend automatically: local Ollama if it is reachable, otherwise the hosted model when `SAATHI_CLOUD_API_KEY` is set, otherwise the offline demo model.

## 0. Accounts and keys (all have free tiers; check current limits)

| service | needed? | what you get |
|---|---|---|
| GitHub | yes | Render deploys from a repo |
| Render | yes | hosting |
| Google AI Studio (aistudio.google.com) | yes | `SAATHI_CLOUD_API_KEY` for hosted **Gemma** |
| Sentry (sentry.io, "Flask" project) | optional | `SENTRY_DSN`: errors + AI call traces |
| ElevenLabs | optional | `ELEVENLABS_API_KEY`: "Listen to the letter" |
| SerpApi | optional | `SERPAPI_API_KEY`: "Also search the web" |

Skip any optional key and that feature simply does not appear.

## 1. Push the code to GitHub

```bash
cd saathi                      # the folder containing render.yaml
git init && git add . && git commit -m "Saathi"
# create an empty repo on github.com, then:
git branch -M main
git remote add origin https://github.com/<you>/saathi.git
git push -u origin main
```
`.gitignore` already excludes `.venv`, `*.db` and `.env`. The GitHub Actions workflow in `.github/workflows/ci.yml` will run the tests on every push.

## 2. Create the service on Render

1. render.com -> **New** -> **Blueprint** -> connect GitHub -> pick the repo. Render reads `render.yaml`.
2. It asks for the secret values (the ones marked `sync: false`). Enter:
   - `SAATHI_PASSWORD`: **required.** Pick a password. Without it, anyone with the URL can read the notes and spend your API credit.
   - `SAATHI_CLOUD_API_KEY`: your Google AI Studio key.
   - `SENTRY_DSN`, `ELEVENLABS_API_KEY`, `SERPAPI_API_KEY`: if you have them (leave blank otherwise).
3. Click **Apply**. The first build takes a few minutes.
4. Open `https://saathi-xxxx.onrender.com`. The browser asks for a login: any username, your `SAATHI_PASSWORD`.

The top-right label should read `gemma-3-27b-it, hosted model`.

### Free tier instead of the paid disk

`render.yaml` uses `plan: starter` because a persistent disk (which keeps the SQLite file between deploys) is a paid feature. For a free demo: change `plan: starter` to `plan: free`, delete the `disk:` block, and delete the `SAATHI_DB` env var. Everything works, but **notes, cards and progress are wiped on each deploy and whenever the free service restarts**, and it sleeps after inactivity so the first request is slow. Fine for a judge's demo, not for the friend you built it for.

## 3. Check it works

1. Profile dialog -> save a name.
2. Notes -> paste `sample_notes/photosynthesis.md` -> Save.
3. Make questions -> Study. A real Gemma question (not "Fill in the blank") means the hosted model is connected.
4. Letter -> Write the letter -> **Listen** (if ElevenLabs is set).
5. Sentry -> Performance / AI Agents: open Explore -> Agents. You should see the agents `Saathi Quiz Writer`, `Saathi Misconception Diagnoser`, `Saathi Notes Assistant` and `Saathi Letter Writer`, each with its model calls, token counts and latency. Prompts and notes are deliberately not sent unless `SENTRY_CAPTURE_PROMPTS=1`.

## 4. Environment variables

| variable | meaning |
|---|---|
| `SAATHI_PASSWORD` | login password for the whole site (strongly recommended) |
| `SAATHI_LLM` | `cloud` on Render. `auto` also works (local Ollama, then cloud, then demo) |
| `SAATHI_CLOUD_API_KEY` | key for the hosted model |
| `SAATHI_CLOUD_MODEL` | default `gemma-3-27b-it` |
| `SAATHI_CLOUD_BASE_URL` | any OpenAI-compatible endpoint (e.g. a DigitalOcean GPU Droplet or Gradient) |
| `SAATHI_CLOUD_JSON_MODE` | set `1` if your model supports `response_format: json_object` |
| `SAATHI_DB` | `/var/data/saathi.db` on the Render disk |
| `SENTRY_DSN` | turns Sentry on |
| `SENTRY_TRACES_SAMPLE_RATE` | `1.0` = keep every trace (0.0 to 1.0) |
| `SENTRY_CAPTURE_PROMPTS` | `1` records prompts and answers too (private notes!). Default: numbers only |
| `SENTRY_ENVIRONMENT` | label like `production` (auto on Render) |
| `SENTRY_SELF_HOSTED` | `1` only if you run your own Sentry server |
| `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL` | voice |
| `SERPAPI_API_KEY` | web search |

## 5. Troubleshooting

- **"Fill in the blank" questions / label says "Demo mode":** the cloud key is missing or `SAATHI_LLM` is wrong. Check the Render logs for `Saathi is using:`.
- **502 from /api/generate:** the model call failed. Check the key, the model name, and your provider's quota. The error text is shown in the app and in Sentry.
- **Few questions generated:** Saathi drops any question that fails its validation. Try again, or set `SAATHI_CLOUD_JSON_MODE=1` if your model supports it.
- **Build fails on Python version:** `render.yaml` pins `PYTHON_VERSION=3.12.7`.
- **Keep one worker.** The SQLite file is written by a single process; the start command uses `--workers 1 --threads 4`. To scale beyond one instance you would move to Postgres or MongoDB Atlas.

## Other platforms (if you prefer)

- **DigitalOcean App Platform / a Droplet:** same start command; on a GPU Droplet you can even keep `SAATHI_LLM=ollama` and run Gemma yourself (privacy-preserving, costs more).
- **Railway / Fly.io:** also fine; set the same env vars and mount a volume for `SAATHI_DB`.
