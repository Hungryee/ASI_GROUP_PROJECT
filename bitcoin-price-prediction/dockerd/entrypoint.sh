#!/bin/sh

yes "$PASSWORD" | passwd > /dev/null 2> /dev/null
yes "$PASSWORD" | passwd "$USERNAME" > /dev/null 2> /dev/null
unset PASSWORD

mkdir /var/run/sshd
/usr/sbin/sshd -o AllowTcpForwarding=yes

cd /app/

uvicorn src.bitcoin_price_prediction.api:app --proxy-headers