#!/usr/bin/env sh

CERTS_DIR="${CERTS_DIR:-/certs}"
CN="${CN:-registry-proxy}"

if [ -d "$CERTS_DIR" ] && ! find "$CERTS_DIR" -mindepth 1 ! -name '.gitignore' | grep -q '.'; then
	# generate certificates according to https://docs.docker.com/registry/insecure
	printf '\n\n\n\n\n%s\n\n' "$CN" \
		| openssl req \
			-newkey rsa:4096 -nodes -sha256 -keyout "$CERTS_DIR"/domain.key \
			-addext 'subjectAltName = DNS:'"$CN" \
			-x509 -days 365 -out "$CERTS_DIR"/domain.crt
fi
