	# Converter

This converter was copied from [X3nOph0N's CityFlow repository](https://github.com/X3nOph0N/CityFlow) as they have implemented fixes so the converter works with the new versions of SUMO.

`converter.py` can convert sumo roadnet files to its corresponding CityFlow version. 

The following code converts a sumo roadnet file, atlanta.net.xml, to CityFlow format.

*Example roadnet and flow files can be downloaded [here](https://github.com/cityflow-project/data/tree/master/tools/Converter/examples)*

```
python converter.py --sumonet atlanta_sumo.net.xml --roadnet atlanta_cityflow.json
```

SUMO roadnet and transformed CityFlow roadnet

<p float="left">
    <img src="https://github.com/cityflow-project/data/raw/master/tools/Converter/figures/sumo.png" alt="SUMO" height="300px"/>
    <img src="https://github.com/cityflow-project/data/raw/master/tools/Converter/figures/city_flow.png" alt="CityFlow" height="300px" style="margin-left:50px"/>
</p>



#### Dependencies

**sumo** is required and make sure that the environment variable *SUMO_HOME* is set properly. If you have an installation via *apt-get*, you can use `/usr/share/sumo` as the value.

**sympy** is required.  You can install it using pip.