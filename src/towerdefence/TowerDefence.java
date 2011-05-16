/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package towerdefence;

import org.newdawn.slick.*;
import org.newdawn.slick.state.StateBasedGame;

public class TowerDefence extends StateBasedGame
{
     static int height = 22*32;
     static int width = 22*32;

     static boolean fullscreen = false;

     static boolean showFPS = true;

     static String title = "Tower Defence";

     static int fpslimit = 60;

     
     public static final int GAMEPLAYSTATE = 1;
//     public static final int PATHTESTSTATE = 2;
     public static final int CUDATESTSTATE = 3;

     

     public TowerDefence(String title)
     {
          super(title);

//          this.addState(new PathTestState(PATHTESTSTATE));
          this.addState(new GameplayState(GAMEPLAYSTATE));
          this.addState(new CudaTestState(CUDATESTSTATE));
          this.enterState(GAMEPLAYSTATE);
//          this.enterState(CUDATESTSTATE);
          //this.enterState(PATHTESTSTATE);
     }

     public static void main(String[] args) throws SlickException
     {
          AppGameContainer app = new AppGameContainer(new ScalableGame(new TowerDefence(title),width,height));

          app.setDisplayMode((int)(width), (int)(height), fullscreen);
          app.setSmoothDeltas(true);
          app.setTargetFrameRate(fpslimit);
          app.setShowFPS(showFPS);
          app.start();
     }

    @Override
    public void initStatesList(GameContainer container) throws SlickException {
        this.getState(GAMEPLAYSTATE).init(container, this);
        this.getState(CUDATESTSTATE).init(container, this);
//        this.getState(PATHTESTSTATE).init(container, this);
    }
}