Task sandbox
############


Authors
-------
Hamza Cherkaoui


Synopsis
--------

Simple package to offer **ACI**, **seed based correlation** and
**seed based GLM** analysis of the task state acquisition done with the PET
MR scanner in Service Hospitalier Frederic Joliot.


Dependencies
------------

* nilearn  
* pypreprocess  


Configuration
-------------

Please edit the *config.py* file to specify the path to the data directory  and
the filename of the data:

Please update your .bashrc by adding

.. code-block:: bash

    export PYTHONPATH="/path/to/task_sandbox/:$PYTHONPATH"

And then source your .bashrc

.. code-block:: bash

    source ~/.bashrc


Instructions
------------

Launch the example:

.. code-block:: bash

    python task.py

You can then inspect the preprocessing report:

.. code-block:: bash

    firefox pypreprocess_output/report_preproc.html

And watch the produce Defautl Mode Network:

.. code-block:: bash

    eog analysis_output/

