"""
Email Parser and Preprocessor Utility
-------------------------------------
Handles parsing of raw email strings, MIME/EML files, HTML content stripping,
and extraction of subject, body, and suspicious email signals.
"""

import email
import re
from email import policy
from html.parser import HTMLParser
from typing import Dict, Any, Optional


class HTMLTextExtractor(HTMLParser):
    """Cleanly strips HTML tags while preserving text and link URLs."""
    def __init__(self):
        super().__init__()
        self.result = []
        self.links = []
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style"):
            self._in_script = True
        elif tag.lower() == "a":
            for attr, val in attrs:
                if attr.lower() == "href" and val:
                    self.links.append(val)

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style"):
            self._in_script = False

    def handle_data(self, data):
        if not self._in_script:
            self.result.append(data)

    def get_text(self) -> str:
        return " ".join("".join(self.result).split())


def clean_html(html_content: str) -> tuple:
    """Strips HTML tags, returning plain text and extracted link hrefs."""
    parser = HTMLTextExtractor()
    parser.feed(html_content)
    return parser.get_text(), parser.links


def parse_email_message(raw_email: str) -> Dict[str, Any]:
    """
    Parses a raw email string or .eml content into a structured dictionary
    and combined text representation for spam classification.
    """
    msg = email.message_from_string(raw_email, policy=policy.default)
    
    subject = msg.get("Subject", "") or ""
    sender = msg.get("From", "") or ""
    recipient = msg.get("To", "") or ""
    date = msg.get("Date", "") or ""

    body_parts = []
    html_parts = []
    html_links = []

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            cdisp = str(part.get("Content-Disposition"))
            if "attachment" in cdisp:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    decoded = payload.decode(charset, errors="replace")
                    if ctype == "text/plain":
                        body_parts.append(decoded)
                    elif ctype == "text/html":
                        html_parts.append(decoded)
            except Exception:
                pass
    else:
        ctype = msg.get_content_type()
        try:
            payload = msg.get_payload(decode=True)
            if payload:
                charset = msg.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
                if ctype == "text/html":
                    html_parts.append(decoded)
                else:
                    body_parts.append(decoded)
            else:
                body_parts.append(msg.get_payload() or "")
        except Exception:
            body_parts.append(str(msg.get_payload() or ""))

    # Prefer plain text, fallback to cleaned HTML
    if body_parts:
        plain_body = " ".join(body_parts)
    elif html_parts:
        plain_body, html_links = clean_html(" ".join(html_parts))
    else:
        plain_body, html_links = clean_html(raw_email)

    # If raw email had no MIME structure, treat raw string as body
    if not subject and not sender and len(plain_body.strip()) == 0:
        plain_body = raw_email

    # Extract suspicious links (e.g. IP addresses, URL shorteners)
    urls = list(set(re.findall(r"https?://[^\s<>\"']+", plain_body) + html_links))
    has_ip_url = any(re.search(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", u) for u in urls)
    has_suspicious_shortener = any(
        any(domain in u.lower() for domain in ["bit.ly", "tinyurl.com", "t.co", "is.gd", "cutt.ly"])
        for u in urls
    )

    # Combined text for model ingestion (include URLs in plain_body if they weren't in text)
    full_text_signals = plain_body + " " + " ".join(urls)
    if subject:
        combined_text = f"Subject: {subject} . {full_text_signals}"
    else:
        combined_text = full_text_signals

    return {
        "subject": subject,
        "from": sender,
        "to": recipient,
        "date": date,
        "body": plain_body.strip(),
        "combined_text": combined_text.strip(),
        "extracted_urls": urls,
        "has_ip_url": has_ip_url,
        "has_suspicious_shortener": has_suspicious_shortener
    }


def parse_eml_file(file_path: str) -> Dict[str, Any]:
    """Reads and parses an .eml file from disk."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    return parse_email_message(content)
