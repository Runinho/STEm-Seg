..
    _ template from https://stackoverflow.com/a/62613202/18724786
    We may want to add :imported-members: to the automodule

{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}
    :members:
    :undoc-members:
    {% if modules %}:imported-members:{% endif %}


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

