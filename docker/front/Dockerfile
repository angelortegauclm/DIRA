FROM nginx:alpine
COPY src/frontend/               /usr/share/nginx/html/
COPY docker/front/40-envsubst.sh /docker-entrypoint.d/40-envsubst.sh
RUN chmod +x /docker-entrypoint.d/40-envsubst.sh
EXPOSE 80
