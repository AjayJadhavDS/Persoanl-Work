library(devtools)
#install_github("twitteR", username="geoffjentry")
library(twitteR)
library(ROAuth)
#install.packages(c("curl", "httr"))
library(curl)

api_key <- "R5B1fEfP8YrY5Qvnntya1pl0h"
api_secret <- "89zubdVwkC3I9NwRPf2yelyQQnWkXrNIdW6GPjErg1jj8vSDzg"
access_token <- "817209686-51p2ZLp6BbOfDAskm6LN0vX5iHCwwosnBvpG83XY"
access_token_secret <- "pKhwb0kZLGafs9l2E5ieAUag8TsuyiLdzn9aWoKWX4wLU"
setup_twitter_oauth(api_key, api_secret, access_token, access_token_secret)

if (interactive()) {
  # table example
  shinyApp(
    ui = fluidPage(
      fluidRow(
        # Enter a hashtag that interests you. 
        column(5,textInput(inputId = "word", 
                           label = "Hashtag", 
                           value='data')),
        
        #Select sample size
        column(5,sliderInput(inputId='n',
                             label='Sample Size',
                             value=20,
                             min=10,
                             max=100))
      ,
        column(12,
               dataTableOutput('table')
        )
      )
    ),
    server = function(input, output) {
    output$table <- renderDataTable(twListToDF(searchTwitter(input$word,input$n)))
    }
  )
}