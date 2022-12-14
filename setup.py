from setuptools import setup, find_packages

setup(
    name='hpcaldb',
    version='0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Geert Jan Talens',
    author_email='gt8538@princeton.edu',
    description='HATPI calibration database package.',
    python_requires='>=3',
    install_requires=['parse>=1.19.0',
                      'astropy>=5.0',
                      'sqlalchemy>=1.4.31']
)
