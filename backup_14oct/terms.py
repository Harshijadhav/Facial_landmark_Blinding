"""
Terms of Use and Acceptable Use Policy configuration for the Proactive Biometric Defense Console.
This file serves as the single source of truth for terms displayed in registration forms and dialogs.
"""

TERMS_VERSION = "1.0"

TERMS_TEXT = """
### 1. Research Project Scope
The **Proactive Biometric Defense Console** is an experimental cyber-forensic research tool developed to study facial landmark obfuscation and evaluate defense resilience against automated deepfake affine-alignment pipelines. It does not provide an absolute guarantee of immunity against all synthetic media or face-recognition models.

### 2. Legal Liability & User Indemnification
By using this platform, you acknowledge and agree that:
- You are solely and personally responsible for any imagery you upload and all downstream usage of the output artifacts.
- If any misuse, non-consensual synthetic generation, defamation, impersonation, or illegal activity is conducted using this platform from the user side, the platform developers and affiliated institutions bear no legal responsibility or liability.
- In the event of valid legal complaints, subpoenas, or official investigations, the platform administrators reserve the right to report and disclose the user's registered identity details, authentication timestamps, and SQLite audit ledger entries to the appropriate legal authorities or affected parties.

### 3. Permitted Usage & Consent
- You must only upload images of yourself or individuals who have provided explicit, informed consent.
- Uploading portraits of minors without verifiable parental or guardian consent is strictly prohibited.
- You agree not to reverse-engineer, disrupt, or overload the application or attempt unauthorized database access.

### 4. Data Retention & Cryptographic Audit Trails
- Image processing is performed strictly in-memory (RAM) and raw uploaded photos are never stored to persistent disk storage.
- For non-repudiation and research accountability, the application maintains a persistent SQLite ledger recording user identifiers, ISO timestamps, output SHA-256 cryptographic image hashes, and quantitative forensic metrics (PSNR, SSIM, alignment shift).
""" 