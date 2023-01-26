#!/bin/sh

yes "$PASSWORD" | passwd > /dev/null 2> /dev/null
yes "$PASSWORD" | passwd "$USERNAME" > /dev/null 2> /dev/null
unset PASSWORD

/usr/sbin/sshd -o AllowTcpForwarding=yes

ARCHFLAGS=-Wno-error=unused-command-line-argument-hard-error-in-future pip install -r requirements.txt

dockerd-entrypoint.sh --host=tcp://127.0.0.1:2375