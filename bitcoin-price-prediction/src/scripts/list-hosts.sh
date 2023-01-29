#!/usr/bin/env sh

docker network inspect ci-cd | jq '.[] | .Containers | to_entries | map({name: .value.Name, host: .value.IPv4Address}) | map(.host |= (split("/") | .[0]))'
