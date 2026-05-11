# ── Dockerfile dira-front ─────────────────────────────────────────────────────
#
# Imagen para el frontend web de DIRA.
# Sirve los ficheros estáticos de src/frontend/ con nginx en el puerto 80.
#
# Al arrancar, el script 40-envsubst.sh sustituye la variable ${API_URL}
# en config.js con el valor de la variable de entorno API_URL inyectada
# por el Deployment de Kubernetes. Así el navegador del usuario sabe a qué
# dirección enviar las peticiones de inferencia sin recompilar la imagen.
#
# Accesible en: http://<node-ip>:30100  (NodePort configurado en values.yaml)

# nginx:alpine es la imagen más ligera para servir contenido estático (~25 MB)
FROM nginx:alpine

# Copiar todos los ficheros del frontend al directorio raíz de nginx
COPY src/frontend/ /usr/share/nginx/html/

# Script que usa envsubst para sustituir ${API_URL} en config.js al arrancar.
# Se coloca en docker-entrypoint.d/ para que nginx lo ejecute automáticamente
# antes de iniciar el servidor (mecanismo estándar de la imagen nginx oficial).
COPY docker/front/40-envsubst.sh /docker-entrypoint.d/40-envsubst.sh
RUN chmod +x /docker-entrypoint.d/40-envsubst.sh

EXPOSE 80
