# DefectVision Frontend

Next.js dashboard for the DefectVision inspection system. See the
[repo root README](../README.md) for the full project (dataset, model, API,
architecture, deployment).

## Development

```bash
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_BASE_URL
npm install
npm run dev
```

Requires the backend API running (see `../backend/`) for any page to show real
data — every page renders an explicit error/empty state rather than mock data
when the API is unreachable or has none yet.
