FROM registry:2.8.1

RUN apk update \
	&& apk add --no-cache \
		openssl

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
