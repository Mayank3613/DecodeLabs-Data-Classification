# Security Policy

## Supported Versions

The following versions of the Enhanced Rule-Based AI Chatbot currently receive security updates and support.

| Version | Supported |
|----------|----------|
| Latest Release | ✅ Yes |
| Previous Release | ✅ Yes |
| Older Releases | ❌ No |

Users are encouraged to always use the latest stable version of the project.

---

# Reporting a Vulnerability

The security of this project is taken seriously. If you discover a security vulnerability, please report it responsibly.

## Please Do Not

- Open a public GitHub Issue for security vulnerabilities.
- Disclose vulnerabilities publicly before they have been reviewed.
- Share exploit code publicly before a fix is available.

---

## How to Report

Please contact the project maintainer with the following information:

### Vulnerability Description

Provide a clear explanation of the issue.

### Steps to Reproduce

Describe how the vulnerability can be reproduced.

### Potential Impact

Explain what an attacker could achieve by exploiting the issue.

### Proof of Concept

Include screenshots, logs, or sample code if available.

### Suggested Fix

If possible, provide recommendations for remediation.

---

## Response Timeline

The project aims to follow the response schedule below:

| Action | Target Time |
|----------|----------|
| Initial Response | Within 72 Hours |
| Vulnerability Assessment | Within 7 Days |
| Fix Development | Depending on Severity |
| Security Release | As Soon As Possible |

Please note that response times may vary depending on project availability and complexity.

---

# Security Best Practices

Contributors and users should follow these practices:

## Dependencies

- Keep all dependencies up to date.
- Regularly review package vulnerabilities.
- Use trusted package sources.

Example:

```bash
pip install --upgrade -r requirements.txt
```

---

## Input Validation

All user input should be treated as untrusted.

Developers should:

- Validate user input.
- Sanitize user-provided data.
- Avoid unsafe code execution.

Avoid:

```python
eval(user_input)
```

Prefer:

```python
ast.literal_eval(user_input)
```

when appropriate.

---

## Sensitive Information

Never commit:

- API Keys
- Passwords
- Access Tokens
- Database Credentials
- Private Certificates
- Personal User Data

Use environment variables instead.

Example:

```python
import os

API_KEY = os.getenv("API_KEY")
```

---

## Git Security

Before pushing code:

```bash
git status
```

Verify that sensitive files are not included.

Recommended additions to `.gitignore`:

```text
.env
venv/
__pycache__/
*.log
*.db
```

---

# Dependency Management

Use:

```bash
pip list --outdated
```

to check outdated packages.

Use:

```bash
pip install --upgrade package_name
```

to update packages.

Developers should regularly review dependency security advisories.

---

# Secure Development Guidelines

Contributors should:

- Follow Python security best practices.
- Avoid hardcoded credentials.
- Validate all user inputs.
- Handle exceptions safely.
- Use least-privilege principles.
- Write secure and maintainable code.

---

# Known Security Considerations

The Enhanced Rule-Based AI Chatbot:

- Does not store user data permanently by default.
- Does not require authentication in local mode.
- Does not transmit user conversations externally.
- Processes conversations locally unless external integrations are added.

If future features introduce:

- Databases
- Authentication
- APIs
- Cloud Deployments
- Voice Processing

additional security reviews should be performed.

---

# Responsible Disclosure

We appreciate responsible disclosure of security vulnerabilities.

Researchers who report valid vulnerabilities responsibly will be acknowledged in project release notes unless anonymity is requested.

Please allow sufficient time for investigation and remediation before public disclosure.

---

# Security Updates

Security fixes will be announced through:

- GitHub Releases
- Release Notes
- Project Documentation

Users should regularly check for updates and upgrade when security patches are released.

---

# Contact

For security-related concerns, please contact:

**Project:** DecodeLabs Data Classification

**Maintainer:** Mayank3613

**Email:** mayug3613@gmail.com

**GitHub:** https://github.com/Mayank3613/DecodeLabs-Data-Classification

---

Thank you for helping keep the DecodeLabs Data Classification project secure for everyone.
