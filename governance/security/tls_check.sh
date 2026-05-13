#!/bin/bash
# Script de vérification de conformité TLS 1.2+
# Utilisation: ./tls_check.sh <endpoint_url>

URL=$1
if [ -z "$URL" ]; then
    echo "Usage: ./tls_check.sh <url>"
    exit 1
fi

echo "🔍 Vérification du certificat TLS pour $URL..."

# Vérifier TLS 1.2
echo "Testing TLS 1.2..."
openssl s_client -connect $URL:443 -tls1_2 < /dev/null 2>/dev/null | grep "Protocol"

# Vérifier TLS 1.3
echo "Testing TLS 1.3..."
openssl s_client -connect $URL:443 -tls1_3 < /dev/null 2>/dev/null | grep "Protocol"

echo "✅ Vérification terminée."
