#!/bin/sh
envsubst '${API_URL}' < /usr/share/nginx/html/js/config.js > /tmp/config.js
mv /tmp/config.js /usr/share/nginx/html/js/config.js
