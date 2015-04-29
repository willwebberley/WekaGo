# WekaGo 1.0

A simple Go wrapper around the Weka Java command-line interface supporting basic classification tasks.

The library abstracts the use of ARFF files and allows users to integrate some of Weka's features programmatically and directly within a Go project.

## Installation

To install WekaGo, retrieve it with:

```bash
$ go get github.com/flyingsparx/wekago
```

and then import it into your project:

```go
import "github.com/flyingsparx/wekago"
```

## Dependencies

No further Go libraries are required, but you will need working Go and Java environments.

You will also need the Weka `jar` file, downloadable from [their website](http://www.cs.waikato.ac.nz/ml/weka). Currently WekaGo is configured to look for `weka.jar` in the root of your Go project, and will fail if it cannot find it.

## Using

### Getting started
Most of the functionality provided by WekaGo is achieved through functions exposed by its API and centred around a Weka model. Models can be created using the `NewModel()` function, to which you should pass your desired classifier:

```go
model := wekago.NewModel("bayes.BayesNet")
```


### Model training
To train a model, `model` needs to be supplied with a series of feature instances. Individual features can be created by identifying a feature name, value, and the ARFF representation of the feature data type (e.g. `'real'` `'{true, false}'`, etc.):

```go
feature1 := wekago.NewFeature("rain", "true", "{true, false}")
feature2 := wekago.NewFeature("grass, "wet", "{wet, dry}")

...
```

To identify features that belong together, they need to be added to an instance:

```go
instance1 := wekago.NewInstance()
instance1.AddFeature(feature1)
instance1.AddFeature(feature2)

instance2 := wekago.NewFeature()
...
```

Once you have a series of training feature instances, they can be added directly to the model:

```go
model.AddTrainingInstance(instance1)
model.AddTrainingInstance(instance2)

...
```

Then training the model is simple:
```go
err := model.Train()
```

Any errors will be reported inside `err`, which will be `nil` if all is OK.

Alternatively, if you already have a model you have built before then you can load that:

```go
model.LoadModel("/path/to/model")
```


### Model testing
Adding test features is almost exactly the same as adding training features, except, if the value of the outcome feature (the last one added to an instance) is unknown, then replace its value with a `"?"`. Then use `model`'s `AddTestingInstance()` method to add instances of such feature sets:

```go
test_feature1 := wekago.NewFeature("rain", "true", "{true, false}")
test_feature2 := wekago.NewFeature("grass", "?", "{wet, dry}")

...

test_instance1 := wekago.NewInstance()
test_instance1.AddFeature(test_feature1)
test_instance1.AddFeature(test_feature2)

...

model.AddTestingInstance(test_instance1)

...
```

Finally, the feature instances can be classified by calling `Test()`:

```go
err := model.Test()
```

As before, any errors will be included in `err`.

Testing the model populates its `Predictions` array.



### Examining classification output
Once you've classified your test set, you can examine the predictions made by your chosen classifier through your `model`'s `Prediction` slice:

```go
for _, prediction := range model.Predictions{
    fmt.Printf("%s\n", prediction.Predicted_value)
}
```

In addition to `Predicted_value`, there are other fields are available in this struct:

* `Index` - the number of the prediction corresponding to the position of the corresponding input test eature instance
* `Observed_value` - the observed value of the corresponding test instance (equal to `"?"` where this was unknown)
* `Predicted_value` - the predicted value of this instance
