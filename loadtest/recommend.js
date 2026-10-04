// Constant-arrival-rate load test for the recommendation endpoint.
// Users come from /data/loadtest_users.json (written by the serving build); 10% of requests
// use IDs that do not exist, which exercises the popularity fallback.
import http from 'k6/http';
import { check } from 'k6';

const users = JSON.parse(open('/data/loadtest_users.json'));
const rate = Number(__ENV.RATE || 100);

export const options = {
  scenarios: {
    warmup: {
      executor: 'constant-arrival-rate', rate: Math.max(1, Math.floor(rate / 4)), timeUnit: '1s',
      duration: '20s', preAllocatedVUs: 20, maxVUs: 100, tags: { phase: 'warmup' },
    },
    load: {
      executor: 'constant-arrival-rate', rate: rate, timeUnit: '1s',
      duration: __ENV.DURATION || '2m', startTime: '20s', preAllocatedVUs: 50, maxVUs: 300,
      tags: { phase: 'load' },
    },
  },
  summaryTrendStats: ['avg', 'p(50)', 'p(95)', 'p(99)', 'max'],
  thresholds: {
    'http_req_duration{phase:load}': ['p(99)<50'],
    'http_req_failed{phase:load}': ['rate<0.01'],
    'dropped_iterations{phase:load}': ['count==0'],
  },
};

export default function () {
  const known = Math.random() >= 0.1;
  const id = known ? users[Math.floor(Math.random() * users.length)] : 10000000 + Math.floor(Math.random() * 1000);
  const res = http.get(`${__ENV.BASE_URL}/recommendations/${id}?k=10`);
  check(res, {
    'status 200': (r) => r.status === 200,
    '10 items': (r) => r.json('items').length === 10,
  });
}

export function handleSummary(data) {
  return { '/data/load_summary.json': JSON.stringify(data, null, 2), stdout: '\n' };
}
