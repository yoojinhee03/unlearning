# 현재 브랜치 변경 사항 파악

Show me what has changed in the current branch compared to main:
1. List all modified files
2. Summarize what each change does
3. Identify any potential issues or conflicts
4. Is there anything that looks unfinished?

Run: git diff main...HEAD --stat && git log main..HEAD --oneline
