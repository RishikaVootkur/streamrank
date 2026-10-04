{{- define "streamrank.labels" -}}
app.kubernetes.io/part-of: streamrank
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
