#!/bin/bash
# Pre-commit credential scanner
# Scans staged files for common secret patterns.
# Install: ln -sf ~/.purple/hooks/scan-secrets.sh .git/hooks/pre-commit
# Or run standalone: ~/.purple/hooks/scan-secrets.sh [file ...]

set -euo pipefail

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

FOUND=0

# Patterns: name, regex
PATTERNS=(
    "AWS Access Key:AKIA[0-9A-Z]{16}"
    "AWS Secret Key:aws.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]"
    "GitHub Token:gh[pousr]_[A-Za-z0-9_]{36,255}"
    "GitHub Classic Token:ghp_[A-Za-z0-9]{36}"
    "Anthropic API Key:sk-ant-[a-zA-Z0-9-]{80,}"
    "OpenAI API Key:sk-[a-zA-Z0-9]{20,}"
    "Azure Tenant ID:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    "Private Key:-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
    "Password Assignment:password\s*[=:]\s*['\"][^'\"]{4,}"
    "Bearer Token:bearer\s+[a-zA-Z0-9._-]{20,}"
    "Basic Auth:basic\s+[a-zA-Z0-9+/=]{20,}"
    "Slack Token:xox[bpors]-[0-9A-Za-z-]{10,}"
    "Stripe Key:sk_live_[0-9a-zA-Z]{24,}"
    "SendGrid Key:SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}"
)

# Allowlisted patterns (false positives)
ALLOWLIST=(
    "ac417330-da1e-4c08-930e-006efc41c47a"
    "example\.com"
    "CHANGE_ME"
    "\.env\.example"
    "placeholder"
    "test.*test"
)

scan_file() {
    local file="$1"
    local file_found=0

    # Skip binary files
    if file -b --mime-type "$file" 2>/dev/null | grep -qv "text/"; then
        return 0
    fi

    for pattern_entry in "${PATTERNS[@]}"; do
        local name="${pattern_entry%%:*}"
        local regex="${pattern_entry#*:}"

        while IFS= read -r match; do
            [ -z "$match" ] && continue

            # Check allowlist
            local allowed=0
            for allow in "${ALLOWLIST[@]}"; do
                if echo "$match" | grep -qiE "$allow"; then
                    allowed=1
                    break
                fi
            done
            [ "$allowed" -eq 1 ] && continue

            if [ "$file_found" -eq 0 ]; then
                echo -e "${RED}>> $file${NC}"
                file_found=1
            fi
            echo -e "  ${YELLOW}[$name]${NC} $match"
            FOUND=$((FOUND + 1))
        done < <(grep -nEi "$regex" "$file" 2>/dev/null || true)
    done
}

# Determine files to scan
if [ $# -gt 0 ]; then
    # Standalone mode: scan specified files
    FILES=("$@")
elif git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    # Pre-commit mode: scan staged files
    FILES=()
    while IFS= read -r f; do FILES+=("$f"); done < <(git diff --cached --name-only --diff-filter=ACM 2>/dev/null)
else
    echo -e "${YELLOW}Not in a git repo and no files specified.${NC}"
    echo "Usage: $0 [file ...]"
    exit 0
fi

if [ ${#FILES[@]} -eq 0 ]; then
    exit 0
fi

echo -e "${GREEN}Scanning ${#FILES[@]} files for secrets...${NC}"

for file in "${FILES[@]}"; do
    [ -f "$file" ] && scan_file "$file"
done

if [ "$FOUND" -gt 0 ]; then
    echo ""
    echo -e "${RED}Found $FOUND potential secret(s). Review before committing.${NC}"
    echo -e "${YELLOW}If these are false positives, add patterns to ALLOWLIST in this script.${NC}"
    exit 1
else
    echo -e "${GREEN}No secrets detected.${NC}"
    exit 0
fi
