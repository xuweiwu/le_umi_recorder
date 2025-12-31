"""
Setup script for quest-controller-client package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / 'README.md'
long_description = readme_file.read_text() if readme_file.exists() else ''

setup(
    name='quest-controller-client',
    version='1.0.0',
    author='Quest Controller Tracking Project',
    description='Python client library for Quest 3 controller tracking',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/controller_tracking_webxr',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.8',
    install_requires=[
        'aiohttp>=3.9.0',
        'websockets>=12.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-asyncio>=0.21',
            'black>=23.0',
            'mypy>=1.0',
        ],
    },
    keywords='quest vr tracking webxr controller',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/controller_tracking_webxr/issues',
        'Source': 'https://github.com/yourusername/controller_tracking_webxr',
    },
)
