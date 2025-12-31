"""
SSL Certificate Generator for Local Development
Generates self-signed certificates for HTTPS support (required for WebXR)
"""

import os
import socket
import ipaddress
from pathlib import Path
from datetime import datetime, timedelta

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID, ExtensionOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
except ImportError:
    print("ERROR: cryptography library not installed")
    print("Install with: pip install cryptography")
    exit(1)


def get_local_ip():
    """Get the local IP address of this machine"""
    try:
        # Create a socket to find the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"


def generate_certificate(output_dir: Path, ip_address: str = None):
    """
    Generate a self-signed SSL certificate

    Args:
        output_dir: Directory to save certificate files
        ip_address: IP address to include in certificate (auto-detected if not provided)
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get IP address
    if ip_address is None:
        ip_address = get_local_ip()

    print(f"Generating SSL certificate for IP: {ip_address}")

    # Generate private key
    print("  Generating private key...")
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    # Certificate subject and issuer
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Local"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Quest Controller Tracking"),
        x509.NameAttribute(NameOID.COMMON_NAME, ip_address),
    ])

    # Build certificate
    print("  Building certificate...")
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address(ip_address)),
            ]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    # Write private key
    key_file = output_dir / "key.pem"
    print(f"  Writing private key: {key_file}")
    with open(key_file, "wb") as f:
        f.write(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )

    # Write certificate
    cert_file = output_dir / "cert.pem"
    print(f"  Writing certificate: {cert_file}")
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print("\n✓ SSL certificate generated successfully!")
    print(f"\n  Certificate: {cert_file}")
    print(f"  Private Key: {key_file}")
    print(f"  Valid for: 365 days")
    print(f"  IP Address: {ip_address}")
    print(f"\n  HTTPS URL: https://{ip_address}:8000")
    print(f"\nNote: You'll need to accept the security warning in your browser")
    print(f"      since this is a self-signed certificate.")

    return cert_file, key_file


def certificates_exist(output_dir: Path) -> bool:
    """Check if certificate files already exist"""
    cert_file = output_dir / "cert.pem"
    key_file = output_dir / "key.pem"
    return cert_file.exists() and key_file.exists()


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate self-signed SSL certificate for local development"
    )
    parser.add_argument(
        "--ip",
        help="IP address to include in certificate (auto-detected if not provided)",
    )
    parser.add_argument(
        "--output",
        default="certs",
        help="Output directory for certificate files (default: certs)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if certificates exist",
    )

    args = parser.parse_args()

    output_dir = Path(__file__).parent / args.output

    # Check if certificates already exist
    if certificates_exist(output_dir) and not args.force:
        print(f"✓ SSL certificates already exist in {output_dir}")
        print(f"\n  Certificate: {output_dir / 'cert.pem'}")
        print(f"  Private Key: {output_dir / 'key.pem'}")
        print(f"\nUse --force to regenerate")
        return

    # Generate new certificates
    try:
        generate_certificate(output_dir, args.ip)
    except Exception as e:
        print(f"\n✗ Error generating certificate: {e}")
        exit(1)


if __name__ == "__main__":
    main()
