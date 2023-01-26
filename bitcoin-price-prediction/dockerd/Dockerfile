FROM docker:dind

RUN apk update \
	&& apk add --no-cache \
		openssh-server \
		openssh-sftp-server \
        gcc g++ make automake python3-dev\
        shadow \
            sudo \
            py-pip\
    && ssh-keygen -A

RUN apk add --no-cache python3


COPY entrypoint.sh /tmp-entrypoint.sh
RUN tr -d '\r' < tmp-entrypoint.sh > entrypoint.sh && rm tmp-entrypoint.sh
RUN chmod +x entrypoint.sh

COPY requirements.txt /

ARG USERNAME
RUN useradd -m "$USERNAME" \
	&& echo "$USERNAME"' ALL=(ALL:ALL) ALL' >> /etc/sudoers

ENV USERNAME="$USERNAME"

ENTRYPOINT ["/entrypoint.sh"]
