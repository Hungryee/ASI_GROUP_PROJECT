FROM python:3.8-slim

#RUN apk update \
#	&& apk add --no-cache \
#		openssh-server \
#		openssh-sftp-server \
#        gcc g++ make automake python3-dev\
#        shadow \
#        sudo \
#        py3-pip\
#    && ssh-keygen -A

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y gcc python3-dev sshpass
COPY /dockerd/entrypoint.sh /app/tmp-entrypoint.sh
RUN tr -d '\r' < /app/tmp-entrypoint.sh > /app/entrypoint.sh && rm /app/tmp-entrypoint.sh
RUN chmod +x /app/entrypoint.sh

COPY /dockerd/requirements.txt /app/

RUN pip3 install --upgrade pip && pip3 install -r /app/requirements.txt
COPY . /app

ARG USERNAME
RUN useradd -m "$USERNAME" \
	&& echo "$USERNAME"' ALL=(ALL:ALL) ALL' >> /etc/sudoers

ENV USERNAME="$USERNAME"

ENTRYPOINT ["/app/entrypoint.sh"]
