# StreamRank

A two-stage movie recommender that serves the next 10 movies for a user in under 50 ms at p99 on a laptop. A PyTorch two-tower model retrieves candidates from the full catalog through an approximate nearest neighbor index, and a LightGBM LambdaMART ranker orders them using long-term history and real-time session features.

Status: work in progress. See [docs/ROADMAP.md](docs/ROADMAP.md) for milestones and results.

## Development

Requires [uv](https://docs.astral.sh/uv/), Docker, and GNU Make.

```bash
make setup     # create the virtual environment and install git hooks
make lint typecheck test
make up        # start local services
make down
```

Copy `.env.example` to `.env` and set local passwords before `make up`.

The project uses MovieLens 32M from GroupLens, downloaded locally. The data is not redistributed in this repository; tests use a synthetic generator with the same schema.

## License

MIT
