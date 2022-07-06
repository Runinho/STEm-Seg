..
    _ template from https://stackoverflow.com/a/62613202/18724786

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:


{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}
