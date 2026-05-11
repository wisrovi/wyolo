#!/bin/bash

# Leer variables de entorno (con valores predeterminados)
CONTROL_HOST=${CONTROL_HOST:-localhost}
CIFS_USER=${CIFS_USER:-wisrovi}
CIFS_PASS=${CIFS_PASS:-wyoloservice}

# Imprimir variables de entorno para depuración
echo
echo "Credenciales samba"
echo "control_server_HOST: $CONTROL_HOST"
echo "USER: $CIFS_USER"
echo "PASS: $CIFS_PASS"

# Instalar paquetes necesarios si no están disponibles
if ! command -v mount.cifs &> /dev/null; then
    echo "Instalando cifs-utils..."
    apt-get update && apt-get install -y cifs-utils
fi

echo
echo "Montando /wyolo/control_server ..."

FOLDER_MOUNT_DATASETS=/wyolo/control_server/datasets
FOLDER_MOUNT_CONFIG=/wyolo/control_server/config_versions
FOLDER_MOUNT_DB=/wyolo/control_server/api_database

# Crear directorio de montaje
mkdir -p $FOLDER_MOUNT_DATASETS
mkdir -p $FOLDER_MOUNT_CONFIG
mkdir -p $FOLDER_MOUNT_DB

umount $FOLDER_MOUNT_DATASETS
umount $FOLDER_MOUNT_CONFIG
umount $FOLDER_MOUNT_DB

chmod 777 -R $FOLDER_MOUNT_DATASETS
chmod 777 -R $FOLDER_MOUNT_CONFIG
chmod 777 -R $FOLDER_MOUNT_DB

# Probar diferentes métodos de montaje que usa Nautilus
echo "Intentando montaje en control_server"

# Probar diferentes configuraciones de montaje
echo "Probando configuración 1: vers=3.0"
# MOUNT_OPTS="username=$CIFS_USER,password=$CIFS_PASS,port=23449,file_mode=0777,dir_mode=0777,iocharset=utf8,uid=$(id -u),gid=$(id -g),vers=3.0,soft"
MOUNT_OPTS="username=$CIFS_USER,password=$CIFS_PASS,port=23449,file_mode=0777,dir_mode=0777,iocharset=utf8"

# Montar cada share con manejo de errores
echo "Montando datasets en $FOLDER_MOUNT_DATASETS..."
mount -t cifs //$CONTROL_HOST/datasets $FOLDER_MOUNT_DATASETS -o $MOUNT_OPTS || echo "Error montando datasets"

echo "Montando config_versions en $FOLDER_MOUNT_CONFIG..."
mount -t cifs //$CONTROL_HOST/config_versions $FOLDER_MOUNT_CONFIG -o $MOUNT_OPTS || echo "Error montando config_versions"

echo "Montando api_database en $FOLDER_MOUNT_DB..."
mount -t cifs //$CONTROL_HOST/api_database $FOLDER_MOUNT_DB -o $MOUNT_OPTS || echo "Error montando api_database"

# echo
# echo "Contenido de los montajes hechos:"
# ls -la $FOLDER_MOUNT_DATASETS
# echo
# ls -la $FOLDER_MOUNT_CONFIG
# echo
# ls -la $FOLDER_MOUNT_DB

# poner la config para dvc en /app/dvc/config
mkdir -p /app/.dvc
cat << EOF > /app/.dvc/config
[core]
    autostage = true
    remote = minio

[remote "control_server"]
    url = s3://dvcstorage
    endpointurl = //$CONTROL_HOST:23444
    access_key_id = $CIFS_USER
    secret_access_key = $CIFS_PASS
    jobs = 8
    read_timeout = 300
    connect_timeout = 60
EOF

