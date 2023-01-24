#version 430 core

in layout(location=0) vec3 position;
in layout(location=1) vec4 color;

out vec4 vertex_color;

//Task 4
uniform layout(location=5) mat4 transform_matrix;

void main()
{
    //Task 2
    //gl_Position = vec4(position, 1.0f);
    //vertex_color = color;

    //Task 3
    //Define transformation_matrix as an identity matrix
    //mat4x4 transform_matrix = mat4(0);
    //int i, j;
    //for(i = 0; i < 4; i++)
    //    for(j=0;j < 4; j++)
    //        transform_matrix[i][j] = i==j ? 1.0 : 0.0;

    //Individually address cells by transform_matrix[column][row], n.b. column major
    // [a b 0 c]
    // [d e 0 f]
    // [0 0 1 0]
    // [0 0 0 1]
    //transform_matrix[0][0] = 1; //a scaling x axis
    //transform_matrix[1][0] = 0; //b tilt/shear x axis
    //transform_matrix[3][0] = 0; //c translate x axis
    //transform_matrix[0][1] = 0; //d tilt/shear x axis
    //transform_matrix[1][1] = 1; //e scaling y axis
    //transform_matrix[3][1] = 0; //f translate y axis

    gl_Position = transform_matrix * vec4(position, 1.0f);
    vertex_color = color;

}