package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"io"
	"log"
	"os"
	"strings"

	"github.com/anthonynsimon/bild/adjust"
	"github.com/disintegration/imaging"
	"github.com/eaciit/gocr"
	"github.com/spf13/cobra"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	resize float32
	input  string
	output string
)

var rootCmd = &cobra.Command{
	Use:   "gotcha",
	Short: "gotcha is a tool for recognizing simple captcha image files.",
	Long:  `Please provide subcommand to take further actions.`,
	Run: func(cmd *cobra.Command, args []string) {
		img := loadImg()
		img = preprocess(img)
		// val := solve(img)
		val := trySolve(img)
		writeFile(val)
	},
}

func init() {
	rootCmd.Flags().StringVarP(&input, "input", "i", "",
		"specify the file path for the captcha image.")
	rootCmd.Flags().StringVarP(&output, "output", "o", "output.txt",
		"specify the file path for the output.")
	rootCmd.Flags().Float32VarP(&resize, "resize", "r", 2,
		"the resize factor for the input captcha image.")

	rootCmd.MarkFlagRequired("input")
}

func main() {
	if e := rootCmd.Execute(); e != nil {
		panic(e)
	}
}

func preprocess(img image.Image) image.Image {
	b := img.Bounds()
	img = adjust.Brightness(img, 0.8)
	img = adjust.Contrast(img, 0.7)
	img = imaging.Resize(img, int(float32(b.Dx())*resize), int(float32(b.Dy())*resize), imaging.Lanczos)
	// img = transform.Resize(img, b.Dx()*f, b.Dy()*f, transform.Gaussian)
	tmp := "tmp.jpeg"
	writeImg(img, tmp)
	return img
}

func trySolve(img image.Image) (val string) {
	var e error
	val, e = solveOCR(img, 64)
	if e == nil {
		return val
	}
	return
}

//writeFile writes string into file.
func writeFile(val string) {
	log.Printf("saving output to file %s", output)
	fo, e := os.Create(output)
	if e != nil {
		log.Panic("failed to create output file ", e)
	}
	defer fo.Close()

	_, e = io.Copy(fo, strings.NewReader(val))
	if e != nil {
		log.Panic("failed to write text "+val+" to file "+output+" ", e)
	}
}

func solveOCR(img image.Image, f int) (val string, e error) {
	defer func() {
		if r := recover(); r != nil {
			hasError := false
			if e, hasError = r.(error); hasError {
				log.Printf("failed to solve: %+v", e)
			}
		}
	}()

	d, _ := os.Getwd()

	// image, _ := gocr.ReadImage(d + "/tmp.jpeg")
	s := gocr.NewCNNPredictorFromDir(d + "/tf_model/")

	// Define the image size
	// s.InputHeight, s.InputWidth = height, width
	// b := img.Bounds()
	// f := int(math.Max(float64(b.Dy()/4), float64(b.Dx()/4)))

	s.InputHeight, s.InputWidth = f, f
	log.Printf("image size: %d, %d", s.InputHeight, s.InputWidth)

	vals := gocr.ScanToStrings(s, img)
	log.Printf("length: %d", len(vals))
	val = strings.Join(vals, "")
	// log.Printf("solved text raw: %s", val)
	// r := strings.NewReplacer(".", "", " ", "")
	// val = r.Replace(val)
	log.Printf("solved text: %s", val)
	return val, nil
}

func solve(img *bytes.Buffer) string {
	// load tensorflow model
	savedModel, e := tf.LoadSavedModel("tensorflow_savedmodel_captcha", []string{"serve"}, nil)
	if e != nil {
		log.Panic("failed to load tensorflow model ", e)
	}
	// solve captcha through tensorflow model
	feedsOutput := tf.Output{
		Op:    savedModel.Graph.Operation("CAPTCHA/input_image_as_bytes"),
		Index: 0,
	}
	imgStr := img.String()
	feedsTensor, e := tf.NewTensor(string(imgStr))
	if e != nil {
		log.Panic("failed to crerate tensor from image buffer ", e)
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
		log.Panic("failed to run tensorflow saved model ", e)
	}

	val := captchaText[0].Value().(string)
	log.Printf("solved captcha text: %s", val)
	return val
}

//loadImg loads and preprocess captcha jpeg image.
func loadImg() (img image.Image) {
	// f, e := os.Open(path)
	// if e != nil {
	// 	log.Panic("unable to read image file ", e)
	// }
	// defer f.Close()
	// j, e := jpeg.Decode(f)
	// if e != nil {
	// 	log.Panic("failed to decode jpeg image "+path+" ", e)
	// }
	img, e := gocr.ReadImage(input)
	if e != nil {
		log.Panic("failed to read jpeg image "+input+" ", e)
	}
	return

	// j = imaging.AdjustGamma(j, 1.5)
	// j = imaging.AdjustBrightness(j, 30)
	// j = imaging.AdjustContrast(j, 300)
	// j = imaging.Sharpen(j, 10.0)

	// b := j.Bounds()
	// ng := image.NewGray(b)
	// for y := b.Min.Y; y < b.Dy(); y++ {
	// 	for x := b.Min.X; x < b.Dx(); x++ {
	// 		c := j.At(x, y)
	// 		ng.Set(x, y, c)
	// 	}
	// }
	// ng := imaging.Grayscale(j)

	//thresholding
	// v, e := strconv.Atoi(os.Args[3])
	// v, e := strconv.ParseFloat(os.Args[3], 32)
	// if e != nil {
	// 	log.Panic("threshold argument is not integer ", e)
	// }
	// img = adjust.Brightness(img, 0.99)
	// img = adjust.Contrast(img, 0.8)
	// img = segment.Threshold(img, uint8(v))
	// img = effect.Sharpen(img)
	// grayscale.Threshold(ng, T, 0, 255)

	// img = ng
	// img = imaging.Blur(ng, 0.2)
	// img = effect.EdgeDetection(ng, v)
	// 1. Create a new filter list and add some filters.
	// g := gift.New(
	// 	gift.Threshold(float32(v)),
	// )

	// 2. Create a new image of the corresponding size.
	// dst is a new target image, src is the original image.
	// dst := image.NewRGBA(g.Bounds(j.Bounds()))

	// 3. Use the Draw func to apply the filters to src and store the result in dst.
	// g.Draw(dst, j)
	// img = dst

	// tmp := "tmp.jpeg"
	// writeImg(img, tmp)

	// buf = new(bytes.Buffer)
	// if data, e := ioutil.ReadFile(tmp); e != nil {
	// 	log.Panic("unable to read image file ", e)
	// } else if _, e = buf.Write(data); e != nil {
	// 	log.Panic("unable to buffer image file ", e)
	// }

	// return j

	// img, e = gocr.ReadImage(tmp)
	// if e != nil {
	// 	log.Panic("unable to read tmp image file ", e)
	// }
	// return
}

func writeImg(img image.Image, path string) {
	outputFile, e := os.Create(path)
	if e != nil {
		log.Panic("failed to save tmp image file ", e)
	}
	defer outputFile.Close()
	if e = jpeg.Encode(outputFile, img, &jpeg.Options{Quality: 100}); e != nil {
		log.Panic("failed to encode tmp jpeg file ", e)
	}
}
