# kind cluster for StreamRank. `make kind-up` fills in ARTIFACTS_DIR and DATA_DIR.
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: streamrank
nodes:
  - role: control-plane
    image: kindest/node:v1.37.0
    extraMounts:
      - hostPath: ARTIFACTS_DIR/serving
        containerPath: /mnt/streamrank/serving
        readOnly: true
      - hostPath: DATA_DIR/feast
        containerPath: /mnt/streamrank/feast
        readOnly: true
      - hostPath: ARTIFACTS_DIR/k8s/redis
        containerPath: /mnt/streamrank/redis
    extraPortMappings:
      - containerPort: 30080
        hostPort: 18000
        protocol: TCP
