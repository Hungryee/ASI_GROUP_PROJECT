FROM nginx:1.21.6

COPY nginx.conf nginx.conf.template

ARG PROXIED_HOST=127.0.0.1
ARG PROXIED_PORT=80
ARG SERVER_NAME=nginx-server

ENV PROXIED_HOST=$PROXIED_HOST
ENV PROXIED_PORT=$PROXIED_PORT
ENV SERVER_NAME=$SERVER_NAME

RUN envsubst '$PROXIED_HOST $PROXIED_PORT $SERVER_NAME' < nginx.conf.template > /etc/nginx/nginx.conf \
	&& rm nginx.conf.template
