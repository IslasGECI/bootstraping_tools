# Dummy Transformations

Para usar este repo como plantilla debemos hacer lo siguiente:

1. Presiona el botón verde que dice _Use this template_
1. Selecciona como dueño a la organización IslasGECI
1. Agrega el nombre del nuevo módulo de python
1. Presiona el botón _Create repository from template_
1. Reemplaza `dummy_transformation` por el nombre del nuevo módulo en:
    - `Makefile`
    - `pyproject.toml`
    - `tests\test_transformation.py`
1. Renombra el archivo `dummy_transformation\transformation.py` al nombre del primer archivo del
   nuevo módulo
1. Cambia la descripción del archivo `dummy_transformation\__init__.py`
1. Renombra el directorio `dummy_transformation` al nombre del nuevo módulo
1. Cambia el `codecov_token` del archivo `Makefile`

Los archivos del nuevo módulo los agregarás en la carpeta que antes se llamaba
`dummy_transformations` y las pruebas en la carpeta `tests`.
