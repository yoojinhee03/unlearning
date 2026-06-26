# 성능 병목 분석

Analyze [FILE/FEATURE] for performance issues:
1. Identify operations that could be expensive (DB queries, loops, I/O)
2. Look for N+1 problems or unnecessary iterations
3. Find any missing caching opportunities
4. Check for memory leaks or excessive allocations

Only flag real issues — don't optimize prematurely.
For each issue, estimate impact (high/medium/low) before suggesting a fix.
