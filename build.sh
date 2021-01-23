pip uninstall ragdoll -y
python setup.py bdist_wheel
pip install --upgrade --force-reinstall dist/*.whl
pip show -f ragdoll
