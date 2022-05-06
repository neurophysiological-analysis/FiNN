import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='finn',
    version='0.1.5',
    description='Toolbox for the analysis of electrophysiological data',
    long_description='A Python Toolbox for the analysis of electrophysiological data',
    author='Maximilian Scherer',
    author_email='maximilian.scherer@posteo.net',
    url='https://github.com/VoodooCode14/FiNN',
    download_url='https://github.com/VoodooCode14/FiNN',
    license='GPLv3',
    packages=['finn.basic','finn.cfc','finn.cleansing','finn.file_io','finn.filters',
    'finn.misc', 'finn.sfc', 'finn.statistical', 'finn.visualization'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: GPLv3 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries',
        ],
    python_requires='>=3.8',
    install_requires=[
    	'numpy',
    	'scipy',
    	'lmfit',
    	'PyQt5',
    	'matplotlib',
    	'mne',
    	'rpy2',
    	'scikit-image',
    ],
    include_package_data = True
)
