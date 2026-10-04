# 12. Kubernetes deployment on kind with CPU autoscaling

Date: 2026-10-04

## Status

Accepted

## Context

The API runs in Docker Compose for development (ADR 0009). The project also needs to show the same image running on Kubernetes, scaling out under load, on one laptop: kind inside Docker Desktop with 8 GB of memory, free tools only.

## Options considered

1. Plain manifests applied with kubectl. Simple, but no values to change per environment.
2. A Helm chart. One `values.yaml` for image, resources, replicas, and autoscaling; `helm upgrade` for changes.
3. Kustomize overlays. Fine for patches, weaker for parameters such as thread counts and limits.

For the serving state in the cluster: a Redis StatefulSet that rebuilds state from the pipeline outputs (slow, needs the full Python toolchain in a Job), or a Redis Deployment that starts from a snapshot of the Compose Redis.

## Decision

Option 2, `deploy/helm/streamrank`:

- API Deployment (CPU request 500m, limit 1 CPU, non-root), NodePort Service on 30080 (host port 18000 through the kind config), and an `autoscaling/v2` HPA on CPU at 60% of the request, 1 to 4 replicas, immediate scale-up and a 60 s scale-down window. With autoscaling on, the Deployment leaves `replicas` unset: Helm 4 applies server-side and otherwise conflicts with the HPA on every upgrade.
- Redis Deployment started from `dump.rdb`, copied out of the Compose Redis by `make kind-up` (186,687 users' state and the Feast online store, 685 MB). Saves are off, so the snapshot stays read-only.
- Serving artifacts and the Feast registry reach the pods through kind `extraMounts` and `hostPath` volumes, read-only. This only works on a one-node local cluster, which is the scope here.
- metrics-server v0.9.0 with `--kubelet-insecure-tls` (kind kubelets use self-signed certificates).
- Images go in with `docker save` and `kind load image-archive`: `kind load docker-image` fails with Docker Desktop's containerd image store.
- Probe timeouts of 5 to 10 s. With the 1 s defaults, a pod saturated before scale-out failed its liveness probe and restarted, pushing its load onto the others.

`make kind-up`, `make helm-install`, `make kind-load` (k6 at a constant rate, HPA logged every 15 s), `make kind-down`.

## What load testing changed in the API

The first scale-out tests showed each request costing about five times more CPU once a backlog formed, so the pods never caught up even at 4 replicas. The engine holds the GIL for most of a request, and the default 40-thread pool let dozens of requests contend for it. The API now runs at most `ENGINE_THREADS` requests at once (chart value `api.engineThreads`, default 8) and queues the rest in the event loop. A cap of 2 also fixed the backlog but raised steady-state p99 from 17 to 133 ms, because requests then wait behind Redis round trips.

The same change helps the Compose container (2 CPUs) at 250 requests per second. Back to back on the same, busier host: p99 151 ms with 8 threads against 3.1 s with 40, and 12 against 1,111 dropped iterations. Both are above the 6.8 ms of ADR 0009, which was measured on a quieter host; the final run re-measures it.

## Results

k6 inside the kind network, 150 requests per second for 4 minutes, starting from one replica, a new connection per request (`NO_REUSE=1`; kept-alive connections stay on the pods that existed when they opened, so new replicas get no traffic):

- The HPA went from 1 to 2 to 4 replicas within 45 s of the pod saturating, the backlog cleared about a minute later, and it settled at 3 replicas at about 54% of the CPU request.
- 36,020 requests, no errors, no pod restarts; median 6.1 ms. p99 is 2.5 s, all from the first minute when one pod (about 130 requests per second at its 1 CPU limit) carried the whole load.
- Steady state with 4 replicas at 150 requests per second: p50 5.7 ms, p99 16.5 ms with kept-alive connections; p99 150 to 190 ms with a new connection per request, from connection setup through the NodePort.

## Consequences

- One `values.yaml` holds the deployment's tuning; the same image runs in Compose and on Kubernetes.
- hostPath volumes and a Redis snapshot tie the chart to a one-node local cluster. A shared cluster would need a registry, object storage or a PersistentVolume for artifacts, and a Redis loader Job.
- Scale-up takes about a minute of saturation to react, so step load beyond one pod's capacity sees a latency spike before replicas arrive. With `minReplicas: 2`, a step up to about 260 requests per second (two pods at their limit) should not saturate before scale-out.
- Docker Desktop's 8 GB is tight: the Compose stack has to be stopped while the cluster runs, or memory pressure pushes per-request time from 8 ms to 40 ms.
