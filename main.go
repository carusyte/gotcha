package main

import (
	"bytes"
	"io"
	"io/ioutil"
	"log"
	"os"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	img := loadImg()

	val := solve(img)

	writeFile(val)
}

//writeFile writes string into file.
func writeFile(val string) {
	filepath := "output"
	if len(os.Args) >= 3 {
		filepath = os.Args[2]
	}
	log.Printf("saving text to file %s", filepath)
	fo, e := os.Create(filepath)
	if e != nil {
		log.Panic("failed to create output file", e)
	}
	defer fo.Close()

	_, e = io.Copy(fo, strings.NewReader(val))
	if e != nil {
		log.Panic("failed to write text "+val+" to file "+filepath, e)
	}
}

func solve(img *bytes.Buffer) string {
	// load tensorflow model
	savedModel, e := tf.LoadSavedModel("./tensorflow_savedmodel_captcha", []string{"serve"}, nil)
	if e != nil {
		log.Panic("failed to load tensorflow model", e)
	}
	// solve captcha through tensorflow model
	feedsOutput := tf.Output{
		Op:    savedModel.Graph.Operation("CAPTCHA/input_image_as_bytes"),
		Index: 0,
	}
	feedsTensor, e := tf.NewTensor(string(img.String()))
	if e != nil {
		log.Panic("failed to crerate tensor from image buffer", e)
	}
	feeds := map[tf.Output]*tf.Tensor{feedsOutput: feedsTensor}

	fetches := []tf.Output{
		{
			Op:    savedModel.Graph.Operation("CAPTCHA/prediction"),
			Index: 0,
		},
	}

	captchaText, e := savedModel.Session.Run(feeds, fetches, nil)
	if e != nil {
		log.Panic("failed to run tensorflow saved model", e)
	}

	val := captchaText[0].Value().(string)
	log.Printf("solved captcha text: %s", val)
	return val
}

func loadImg() (img *bytes.Buffer) {
	img = new(bytes.Buffer)
	if data, e := ioutil.ReadFile(os.Args[1]); e != nil {
		log.Panic("unable to read image file", e)
	} else if _, e = img.Write(data); e != nil {
		log.Panic("unable to buffer image file", e)
	}
	return
}
